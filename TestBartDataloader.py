import unittest
from transformers import BartTokenizer
import json

from .utils_focus import get_dataset_only_train_dev, get_dataset_only_test
from .data_utils import build_input_from_segments_bart_inctxt


def get_object(**args):
    return args


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")


class TestDataloaders(unittest.TestCase):
    def test_0_get_dataset_only_train_dev(self):
        # я копирую эти данные из дебагера и как иначе получить объект не знаю
        args = load_json("args.json")
        all_dataset = get_dataset_only_train_dev(
            tokenizer,
            args.train_dataset_path,
            args.train_dataset_cache,
            args.dev_dataset_path,
            args.dev_dataset_cache,
            args.debug,
        )

        mocked_all_dataset = load_json("./all_dataset.json")
        self.assertEqual(all_dataset, mocked_all_dataset)

    def test_1_build_input_from_segments_bart_inctxt(self):
        all_dataset = load_json("./get_dataset_only_train_dev__output.json")
        args = load_json("args.json")
        max_history = 5
        testset = False
        test_instances = []
        for dataset_name, dataset in all_dataset.items():
            for dialog in dataset:
                dialogID = dialog["dialogID"]
                persona = dialog["persona"]
                knowledge = dialog["knowledge"]
                utterance = dialog["utterance"]
                for i, utt in enumerate(utterance):
                    history = utt["dialog"][-(2 * max_history) :]
                    persona_candidates = utt["persona_candidates"]
                    persona_grouding = utt["persona_grounding"]
                    knowledge_candidates = utt["knowledge_candidates"]
                    knowledge_answer_index = utt["knowledge_answer_index"]
                    instance = build_input_from_segments_bart_inctxt(
                        persona,
                        knowledge,
                        history,
                        persona_candidates,
                        persona_grouding,
                        knowledge_candidates,
                        knowledge_answer_index,
                        dialogID,
                        tokenizer,
                        lm_labels=True,
                        testset=testset,
                        inference=args.inference,
                    )
                    test_instances.append(instance)
        mocked_instances = load_json(
            "build_input_from_segments_bart_inctxt__test_instances.json"
        )
        self.assertEqual(test_instances, mocked_instances)
