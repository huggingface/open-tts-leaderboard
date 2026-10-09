"""Offline contract checks for API orchestration and independent result export."""

import base64
import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))
import run_pipeline
import score_results


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.models = ModuleType("api.models")
        self.models.MODELS = {
            "fish/s2-pro": SimpleNamespace(provider="fish", languages=("en", "zh", "fr"), reference_mode="inline"),
            "example/voice": SimpleNamespace(provider="deepgram", languages=("en",), reference_mode=None),
        }
        self.models.get_model = self.models.MODELS.__getitem__
        self.modules_patch = patch.dict(sys.modules, {"api.models": self.models})
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.modules_patch.start()
        self.env_patch.start()
        self.addCleanup(self.modules_patch.stop)
        self.addCleanup(self.env_patch.stop)

    def args(self, *flags):
        return run_pipeline.make_parser().parse_args(["--models", "fish/s2-pro", *flags])

    def test_generation_only_needs_no_bucket(self):
        args = self.args("--stages", "generate", "--max_eval_samples", "8")
        run_pipeline.validate_args(args)
        config = run_pipeline.dataset_configs(args)[0]
        command = run_pipeline.generation_command(args, "fish/s2-pro", config)
        self.assertEqual(command[command.index("--max_eval_samples") + 1], "8")
        self.assertEqual(command[:3], ["docker", "run", "--rm"])
        self.assertIn("HF_TOKEN", command)

    def test_remote_stages_require_separate_explicit_bucket(self):
        with self.assertRaisesRegex(ValueError, "dedicated API"):
            run_pipeline.validate_args(self.args())
        with self.assertRaisesRegex(ValueError, "separate bucket"):
            run_pipeline.validate_args(self.args("--results_bucket", run_pipeline.H200_BUCKET))

    def test_clone_capability_validated_before_generation(self):
        args = run_pipeline.make_parser().parse_args(["--models", "example/voice", "--stages", "generate", "--voice_clone"])
        with self.assertRaisesRegex(ValueError, "no inline"):
            run_pipeline.validate_args(args)

    def test_dataset_configs_match_shared_manifest_names(self):
        args = self.args("--only_langs", "en", "fr")
        configs = run_pipeline.dataset_configs(args)
        self.assertEqual([(c.dataset, c.split, c.language) for c in configs],
                         [("tts", "en", "en"), ("zero_shot", "en", "en"), ("zero_shot", "fr", "fr")])
        path = run_pipeline.manifest_path("fish/s2-pro", configs[0], True)
        self.assertEqual(path.name, "MODEL_fish-s2-pro_DATASET_bezzam-seed_tts_eval_tts_en_voice_clone.jsonl")

    def test_injection_is_portable_and_quotes_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source.py"
            source.write_text("print('hello')\n", encoding="utf-8")
            snippet = run_pipeline.injected_script(source, "/app/path with spaces/script.py")
            tokens = shlex.split(snippet)
            self.assertEqual(tokens[:2], ["python", "-c"])
            self.assertIn(base64.b64encode(source.read_bytes()).decode(), tokens[2])
            self.assertNotIn("base64 -w0", snippet)

    def test_pipeline_orders_generation_upload_jobs_download_score(self):
        args = self.args("--datasets", "seed_tts", "--only_langs", "en", "--results_bucket", "test/api-results",
                         "--voice_clone", "--dry_run")
        with patch.object(run_pipeline, "run_command") as runner:
            run_pipeline.run_pipeline(args)
        commands = [call.args[0] for call in runner.call_args_list]
        self.assertEqual(len(commands), 7)
        self.assertEqual(commands[0][:2], ["docker", "pull"])
        self.assertEqual(commands[1][:3], ["docker", "run", "--rm"])
        self.assertIn("/app/api/run_eval.py", commands[1])
        self.assertEqual(commands[2][:3], ["hf", "buckets", "sync"])
        self.assertEqual(commands[3][:3], ["hf", "jobs", "run"])
        self.assertIn("--asr_language en", commands[3][-1])
        self.assertIn("--overwrite", commands[3][-1])
        self.assertIn("--sim_backend wavlm_seed_tts", commands[4][-1])
        self.assertEqual(commands[5][-2:], ["--exclude", "*.wav"])
        self.assertNotIn("--delete", commands[5])
        self.assertTrue(commands[6][1].endswith("score_results.py"))

    def test_generation_credentials_stay_out_of_logged_arguments(self):
        args = self.args("--stages", "generate")
        with patch.dict(os.environ, {"HF_TOKEN": "hf-test-secret", "FISH_API_KEY": "fish-test-secret",
                                     "DEEPGRAM_API_KEY": "unused-secret"}):
            command = run_pipeline.generation_command(args, "fish/s2-pro", run_pipeline.dataset_configs(args)[0])
        self.assertIn("FISH_API_KEY", command)
        self.assertNotIn("DEEPGRAM_API_KEY", command)
        self.assertFalse(any("secret" in part for part in command))

    def test_generation_mounts_persistent_results_cache_and_host_identity(self):
        args = self.args("--stages", "generate")
        with tempfile.TemporaryDirectory(prefix="tts cache ") as cache, patch.dict(os.environ, {"HF_CACHE_DIR": cache}):
            command = run_pipeline.generation_command(args, "fish/s2-pro", run_pipeline.dataset_configs(args)[0])
        self.assertIn(f"{Path(cache).resolve()}:/hf_cache", command)
        self.assertIn(f"{API_DIR / 'results'}:/app/api/results", command)
        self.assertIn(f"{API_DIR}:/app/api:ro", command)
        self.assertIn(f"{run_pipeline.REPO_ROOT / 'scripts'}:/app/scripts:ro", command)
        self.assertEqual(command[command.index("--entrypoint") + 1], "python")
        self.assertEqual(command[command.index("--workdir") + 1], "/app/api")
        self.assertEqual(command[command.index("--platform") + 1], "linux/amd64")
        self.assertEqual(command[command.index("--user") + 1], f"{os.getuid()}:{os.getgid()}")

    def test_local_fixture_and_external_references_are_readable_in_container(self):
        with tempfile.TemporaryDirectory(prefix="tts fixture ") as folder:
            root = Path(folder).resolve()
            source = root / "samples.jsonl"
            source.write_text(json.dumps({"text": "hello", "prompt_audio_filepath": "refs/prompt.wav"}) + "\n" +
                              json.dumps({"text": "hello", "prompt_audio_filepath": str(root / "external/prompt.wav")}) + "\n")
            args = self.args("--stages", "generate", "--datasets", "seed_tts", "--only_langs", "en",
                             "--input_jsonl", str(source))
            command = run_pipeline.generation_command(args, "fish/s2-pro", run_pipeline.dataset_configs(args)[0])
        for path in (source, root / "refs/prompt.wav", root / "external/prompt.wav"):
            self.assertIn(f"{path}:{path}:ro", command)
        self.assertEqual(command[command.index("--input_jsonl") + 1], str(source))

    def test_fixture_reference_symlinks_preserve_the_container_path_alias(self):
        with tempfile.TemporaryDirectory(prefix="tts symlink ") as folder:
            root = Path(folder).resolve()
            reference = root / "reference.wav"
            reference.write_bytes(b"reference")
            alias = root / "prompt.wav"
            alias.symlink_to(reference)
            source = root / "samples.jsonl"
            source.write_text(json.dumps({"text": "hello", "prompt_audio_filepath": str(alias)}) + "\n")
            args = self.args("--stages", "generate", "--input_jsonl", str(source))
            command = run_pipeline.generation_command(args, "fish/s2-pro", run_pipeline.dataset_configs(args)[0])
        self.assertIn(f"{reference}:{alias}:ro", command)

    def test_dry_run_pulls_once_across_models_and_writes_no_directories(self):
        args = run_pipeline.make_parser().parse_args(["--models", "fish/s2-pro", "example/voice", "--stages", "generate",
                                                     "--datasets", "seed_tts", "--only_langs", "en", "--dry_run"])
        with patch.object(run_pipeline, "run_command") as runner, patch.object(Path, "mkdir") as mkdir:
            run_pipeline.run_pipeline(args)
        commands = [call.args[0] for call in runner.call_args_list]
        self.assertEqual([command[:2] for command in commands], [["docker", "pull"], ["docker", "run"], ["docker", "run"]])
        mkdir.assert_not_called()

    def test_cached_image_skips_pull_and_uses_exact_digest(self):
        image = "registry.hf.space/example-env@sha256:" + "a" * 64
        args = self.args("--stages", "generate", "--datasets", "seed_tts", "--only_langs", "en", "--dry_run",
                         "--skip_image_pull", "--api_image", image)
        with patch.object(run_pipeline, "run_command") as runner:
            run_pipeline.run_pipeline(args)
        self.assertEqual(runner.call_count, 1)
        self.assertIn(image, runner.call_args.args[0])

    def test_default_and_custom_spaces_resolve_to_hub_registry_images(self):
        args = self.args("--stages", "generate")
        self.assertEqual(run_pipeline.generation_image(args), "registry.hf.space/hf-audio-open-tts-leaderboard-apis:latest")
        args.tts_space = "Example/tts_api"
        self.assertEqual(run_pipeline.generation_image(args), "registry.hf.space/example-tts-api:latest")
        args.tts_space = "invalid-space"
        with self.assertRaisesRegex(ValueError, "exact reference"):
            run_pipeline.validate_args(args)

    def test_registry_failure_builds_from_the_exact_hub_space_commit(self):
        args = self.args("--stages", "generate")
        info = SimpleNamespace(sdk="docker", private=False, sha="a" * 40)
        failure = run_pipeline.subprocess.CalledProcessError(1, ["docker", "pull"])
        with (patch.object(run_pipeline.shutil, "which", return_value="docker"), patch.object(Path, "mkdir"),
              patch("huggingface_hub.HfApi") as hub, patch.object(run_pipeline, "run_command", side_effect=[failure, None, None]) as runner):
            hub.return_value.space_info.return_value = info
            run_pipeline.prepare_generation(args)
        hub.return_value.space_info.assert_called_once_with(run_pipeline.DEFAULT_TTS_SPACE)
        download, build = [call.args[0] for call in runner.call_args_list[1:]]
        self.assertEqual(download[:3], ["hf", "download", run_pipeline.DEFAULT_TTS_SPACE])
        self.assertEqual(download[download.index("--revision") + 1], info.sha)
        self.assertEqual(build[:2], ["docker", "build"])
        self.assertEqual(Path(build[-1]).name, info.sha)
        self.assertEqual(build[build.index("--platform") + 1], "linux/amd64")
        self.assertEqual(build[build.index("--tag") + 1], run_pipeline.generation_image(args))

    def test_explicit_image_pull_failure_does_not_substitute_a_space_build(self):
        args = self.args("--stages", "generate", "--api_image", "registry.example/env@sha256:" + "a" * 64)
        failure = run_pipeline.subprocess.CalledProcessError(1, ["docker", "pull"])
        with (patch.object(run_pipeline.shutil, "which", return_value="docker"), patch.object(Path, "mkdir"),
              patch.object(run_pipeline, "run_command", side_effect=failure),
              patch.object(run_pipeline, "build_space_environment") as fallback,
              self.assertRaises(run_pipeline.subprocess.CalledProcessError)):
            run_pipeline.prepare_generation(args)
        fallback.assert_not_called()

    def test_missing_docker_fails_before_generation_or_directory_creation(self):
        args = self.args("--stages", "generate")
        with (patch.object(run_pipeline.shutil, "which", return_value=None), patch.object(Path, "mkdir") as mkdir,
              self.assertRaisesRegex(ValueError, "Docker is required")):
            run_pipeline.run_pipeline(args)
        mkdir.assert_not_called()

    def test_remote_only_downloads_bucket_manifests_without_uploading_stale_local_state(self):
        args = self.args("--datasets", "seed_tts", "--only_langs", "en", "--results_bucket", "test/api-results",
                         "--stages", "transcribe", "score", "--dry_run")
        with patch.object(run_pipeline, "run_command") as runner:
            run_pipeline.run_pipeline(args)
        commands = [call.args[0] for call in runner.call_args_list]
        syncs = [command for command in commands if command[:3] == ["hf", "buckets", "sync"]]
        self.assertEqual(len(syncs), 2)
        self.assertTrue(all(command[3].startswith("hf://buckets/") for command in syncs))
        self.assertFalse(any(command[0] == "docker" for command in commands))

    def test_full_split_ttfa_works_only_without_scorer_stages(self):
        args = self.args("--stages", "generate", "--ttfa_probe=-1")
        run_pipeline.validate_args(args)
        with self.assertRaisesRegex(ValueError, "sidecar only"):
            run_pipeline.validate_args(self.args("--ttfa_probe=-1"))

    def test_overwrite_forwarded_to_generation(self):
        args = self.args("--stages", "generate", "--overwrite")
        config = run_pipeline.dataset_configs(args)[0]
        self.assertIn("--overwrite", run_pipeline.generation_command(args, "fish/s2-pro", config))

    def test_local_fixture_not_relabelled_across_splits(self):
        args = self.args("--stages", "generate", "--input_jsonl", "samples.jsonl")
        with self.assertRaisesRegex(ValueError, "avoid relabelling"):
            run_pipeline.validate_args(args)

    def test_zero_workers_and_unknown_language_rejected(self):
        with self.assertRaisesRegex(ValueError, "max_workers"):
            run_pipeline.validate_args(self.args("--stages", "generate", "--max_workers", "0"))
        with self.assertRaisesRegex(ValueError, "language codes"):
            run_pipeline.validate_args(self.args("--stages", "generate", "--only_langs", "xx"))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.calls = []

    def manifest(self, language="en", clone=False):
        path = self.directory / f"MODEL_fish-s2-pro_DATASET_test-eval_tts_{language}{'_voice_clone' if clone else ''}.jsonl"
        rows = [
            {"audio_filepath": f"tts_{language}/output_{index}.wav", "text": "some text", "pred_text": "some text",
             "duration": 1.2, "time": None, "timing_backend": "api", "language": language,
             "model_id": "fish/s2-pro", "api_latency_s": latency, "api_attempts": 1,
             "api_ttfa_ms": 100 + index * 100}
            for index, latency in enumerate((1.0, 3.0))
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        metadata = {"model_id": "fish/s2-pro", "provider": "fish", "model": "s2-pro", "voice": None,
                    "language": language, "dataset_path": "test/eval", "dataset": "tts", "split": language,
                    "voice_clone": clone, "n_samples": 2, "n_completed": 2, "n_failed": 0, "complete": True,
                    "max_workers": 1, "wall_time_s": 4.0, "api_throughput_rtfx": .6}
        path.with_suffix(".run.json").write_text(json.dumps(metadata), encoding="utf-8")
        return path, rows

    def evaluate(self, directory, **kwargs):
        self.calls.append(kwargs)
        return {}, {"fish/s2-pro | test-eval": {"wer": 0.0, "metric": "WER", "sim": 95.0, "rtfx": None}}

    def test_language_scoped_quality_and_separate_api_metrics(self):
        en, _ = self.manifest("en")
        fr, _ = self.manifest("fr")
        rows = score_results.collect_results([en, fr], evaluator=self.evaluate)
        self.assertEqual({call["language"] for call in self.calls}, {"en", "fr"})
        self.assertEqual(rows[0]["api_latency_p50_s"], 2.0)
        self.assertAlmostEqual(rows[0]["api_latency_p95_s"], 2.9)
        self.assertEqual(rows[0]["api_ttfa_p50_ms"], 150)
        self.assertEqual(rows[0]["api_throughput_rtfx"], .6)
        self.assertIsNone(rows[0]["sim"])
        self.assertNotIn("rtfx", rows[0])

    def test_refuses_gpu_time_and_mismatched_language(self):
        path, rows = self.manifest()
        rows[0]["time"] = .5
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "GPU/API"):
            score_results.collect_results([path], evaluator=self.evaluate)
        rows[0]["time"] = None
        rows[0]["language"] = "fr"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "language/model"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_missing_metadata_is_rejected(self):
        path, _ = self.manifest()
        path.with_suffix(".run.json").unlink()
        with self.assertRaisesRegex(ValueError, "metadata"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_stale_sim_fork_is_rejected(self):
        path, rows = self.manifest(clone=True)
        rows[0]["api_latency_s"] = 999
        fork = path.with_name(path.stem + "_wavlm_seed_tts.jsonl")
        fork.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "stale"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_incomplete_counts_and_missing_asr_are_rejected(self):
        path, rows = self.manifest()
        sidecar = path.with_suffix(".run.json")
        metadata = json.loads(sidecar.read_text())
        metadata["complete"] = False
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "incomplete API"):
            score_results.collect_results([path], evaluator=self.evaluate)
        metadata["complete"] = True
        metadata["n_samples"] = 3
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "sample counts"):
            score_results.collect_results([path], evaluator=self.evaluate)
        metadata["n_samples"] = 2
        sidecar.write_text(json.dumps(metadata))
        del rows[0]["pred_text"]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "ASR predictions"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_clone_similarity_required_and_known_short_exclusions_allowed(self):
        path, rows = self.manifest(clone=True)
        with self.assertRaisesRegex(ValueError, "incomplete.*similarity"):
            score_results.collect_results([path], evaluator=self.evaluate)
        model = "wavlm_large_finetune+ecapa_tdnn (wavlm_large_finetune.pth)"
        for row in rows:
            row["prompt_audio_filepath"] = "reference.wav"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        rows[0].update(sim=.95, sim_model=model)
        rows[1].update(sim=None, sim_model=model, sim_note="clip shorter than the 320-sample minimum")
        fork = path.with_name(path.stem + "_wavlm_seed_tts.jsonl")
        fork.write_text("".join(json.dumps(row) + "\n" for row in rows))
        exported = score_results.collect_results([path], evaluator=self.evaluate)
        self.assertEqual(exported[0]["sim"], 95.0)
        rows[0]["pred_text"] = "stale transcription"
        fork.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "stale"):
            score_results.collect_results([path], evaluator=self.evaluate)

    def test_export_merges_evaluated_splits_without_placeholder_models(self):
        en, _ = self.manifest("en")
        fr, _ = self.manifest("fr")
        score_results.write_results(score_results.collect_results([en], evaluator=self.evaluate), self.directory)
        csv_path, json_path = score_results.write_results(score_results.collect_results([fr], evaluator=self.evaluate), self.directory)
        document = json.loads(json_path.read_text())
        self.assertEqual(len(document["results"]), 2)
        self.assertEqual({row["model_id"] for row in document["results"]}, {"fish/s2-pro"})
        self.assertNotIn(",rtfx,", csv_path.read_text().splitlines()[0])
        self.assertEqual(document["timing_backend"], "api")


if __name__ == "__main__":
    unittest.main()
