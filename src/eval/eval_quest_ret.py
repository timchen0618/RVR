# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Analyze retriever predictions."""

from absl import app
from absl import flags
import dataclasses
import typing
import json

import collections


def print_avg_by_key(examples, metrics, key_fn):
  """Prints avg results partitioned according to some `key_fn`.

  Args:
    examples: List of Example instances.
    metrics: List of float values corresponding to `examples`.
    key_fn: Function that takes an Example and returns a key to aggregate over.
  """
  key_to_metrics = collections.defaultdict(list)
  for example, metric in zip(examples, metrics):
    key = key_fn(example)
    key_to_metrics[key].append(metric)
  # Compute averages.
  key_to_avg = {
      key: sum(vals) / len(vals) for key, vals in key_to_metrics.items()
  }
  for key, val in key_to_avg.items():
    print("%s (%s): %s" % (key, len(key_to_metrics[key]), val))


def print_avg_by_template(examples, metrics):
  """Prints results partitioned by template."""
  key_fn = lambda example: example.metadata.template
  return print_avg_by_key(examples, metrics, key_fn)


def print_avg(examples, metrics):
  key_fn = lambda unused_x: "all"
  return print_avg_by_key(examples, metrics, key_fn)


def read_json(file_path):
    "Read json file."
    with open(file_path, 'r') as file:
        return json.load(file)

def read_jsonl(file_path):
  """Read jsonl file to a List of Dicts."""
  with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

  

@dataclasses.dataclass(frozen=False)
class ExampleMetadata:
  """Optional metadata used for analysis."""
  # The template used to synthesize this example.
  template: typing.Optional[str] = None
  # The domain of the example (e.g. films, books, plants, animals).
  domain: typing.Optional[str] = None
  # Fluency labels
  fluency: typing.Optional[typing.Sequence[bool]] = None
  # Meaning labels
  meaning: typing.Optional[typing.Sequence[bool]] = None
  # Naturalness labels
  naturalness: typing.Optional[typing.Sequence[bool]] = None
  # The following fields are dictionaries keyed by document title.
  # The sequences can contain multiple values for replicated annotations.
  relevance_ratings: typing.Optional[typing.Dict[str,
                                                 typing.Sequence[str]]] = None
  evidence_ratings: typing.Optional[typing.Dict[str,
                                                typing.Sequence[str]]] = None
  # The nested value is a map from query substring to document substring.
  attributions: typing.Optional[typing.Dict[str, typing.Sequence[typing.Dict[
      str, str]]]] = None


@dataclasses.dataclass(frozen=False)
class Example:
  """Represents a query paired with a set of documents."""
  query: str
  docs: typing.Iterable[str]
  # TODO(cmalaviya,petershaw): Move this to `metadata`.
  original_query: typing.Optional[str] = None
  # Scores can be optionally included if the examples are generated from model
  # predictions. Indexes of `scores` should correspond to indexes of `docs`.
  scores: typing.Optional[typing.Iterable[float]] = None
  # Optional metadata.
  metadata: typing.Optional[ExampleMetadata] = None


def read_examples(filepath):
  examples_json = read_jsonl(filepath)
  if 'docs' not in examples_json[0]:
    examples_json = [dict(query=ex['question'], docs=[ctx['title'] for ctx in ex['ctxs']]) for ex in examples_json]
  
  examples = [Example(**example) for example in examples_json]
  for example in examples:
    if example.metadata is not None:
        example.metadata = ExampleMetadata(**example.metadata)
  return examples

  


FLAGS = flags.FLAGS

flags.DEFINE_string("gold", "", "Examples jsonl file with gold document set.")

flags.DEFINE_string("pred", "", "Examples jsonl file with predicted documents.")

flags.DEFINE_integer("offset", 0, "Start index for examples to process.")

flags.DEFINE_integer("limit", 0, "End index for examples to process if >0.")

flags.DEFINE_bool("verbose", False, "Whether to print out verbose debugging.")

# Values of k to use to compute MRecall@k.
# K_VALS = [20, 50, 100, 1000]
K_VALS = [20]


def main(unused_argv):
  gold_examples = read_examples(FLAGS.gold)
  pred_examples = read_examples(FLAGS.pred)

  num_examples = 0
  # Dictionary mapping mrecall k value to number of examples where predicted
  # set is superset of gold set.
  mrecall_vals = {k: [] for k in K_VALS}
  # List of recall for each example.
  recall_vals = {k: [] for k in K_VALS}

  query_to_pred_example = {ex.query: ex for ex in pred_examples}
  for idx, gold_example in enumerate(gold_examples):
    if FLAGS.offset and idx < FLAGS.offset:
      continue
    if FLAGS.limit and idx >= FLAGS.limit:
      break

    if not gold_example.docs:
      raise ValueError("Example has 0 docs.")

    if FLAGS.verbose:
      print("\n\nProcessing example %s: `%s`" % (idx, gold_example))
      num_examples += 1

    pred_example = query_to_pred_example[gold_example.query]

    for k in K_VALS:
      if FLAGS.verbose:
        print("Evaluating MRecall@%s" % k)
      predicted_docs = set(pred_example.docs[:k])
      gold_docs = set(gold_example.docs)
      if gold_docs.issubset(predicted_docs):
        if FLAGS.verbose:
          print("Contains all docs!")
        mrecall_vals[k].append(1.0)
      else:
        mrecall_vals[k].append(0.0)

      # Compute recall.
      covered_docs = gold_docs.intersection(predicted_docs)
      recall = float(len(covered_docs)) / len(gold_docs)
      recall_vals[k].append(recall)

      # Print debugging.
      extra_docs = predicted_docs.difference(gold_docs)
      missing_docs = gold_docs.difference(predicted_docs)
      if FLAGS.verbose:
        print("Extra docs: %s" % extra_docs)
        print("Missing docs: %s" % missing_docs)

  print("num_examples: %s" % num_examples)

  for k in K_VALS:
    print("MRecall@%s" % k)
    print_avg(gold_examples, mrecall_vals[k])
    print_avg_by_template(gold_examples, mrecall_vals[k])
    print("Avg. Recall@%s" % k)
    print_avg(gold_examples, recall_vals[k])
    print_avg_by_template(gold_examples, recall_vals[k])


if __name__ == "__main__":
  app.run(main)