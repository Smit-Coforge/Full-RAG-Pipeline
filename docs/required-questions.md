# Required question results

Saved output from `python -m mini_rag_lab evaluate` on 20 September 2026.
All six assignment questions passed.

Reproduce with:

```shell
python -m mini_rag_lab migrate
python -m mini_rag_lab ingest
python -m mini_rag_lab evaluate
```

The same cases are also covered by
`tests/integration/test_live_pipeline.py` when `RUN_LIVE_TESTS=1`.

## Summary

| Question | Result | Citation |
| --- | --- | --- |
| How much can I spend on food each day? | $65 per day | 1. Meals |
| Can I book first-class airfare? | Economy required; business class needs approval | 3. Airfare |
| My hotel costs $250. What do I need? | Manager approval before booking | 2. Hotels |
| Do I need a receipt for a $20 taxi? | No; $20 is below $25 | 5. Receipts |
| Can I claim a limousine upgrade? | Luxury upgrades are not reimbursable | 4. Ground Transportation |
| Does the company reimburse gym memberships? | Exact refusal | none |

Each supported answer retrieved at most three chunks, sorted by ascending
cosine distance, and cited the expected section. The gym-membership question
returned `The provided policy does not answer this question.` with
`"citation": null`.

## Full evaluator output

```json
{
  "passed": true,
  "passed_count": 6,
  "total": 6,
  "results": [
    {
      "question": "How much can I spend on food each day?",
      "passed": true,
      "checks": {
        "expected_section_retrieved": true,
        "expected_section_cited": true,
        "answer_contains_expected_terms": true
      },
      "response": {
        "answer": "Employees may claim up to $65 per day for meals while traveling overnight.",
        "citation": {
          "document": "Employee Expense Policy",
          "version": "2.0",
          "section": "1. Meals"
        },
        "retrieved_chunks": [
          {
            "section": "1. Meals",
            "distance": 0.3720391545248638
          },
          {
            "section": "5. Receipts",
            "distance": 0.47442206892831107
          },
          {
            "section": "2. Hotels",
            "distance": 0.5341998216391215
          }
        ]
      }
    },
    {
      "question": "Can I book first-class airfare?",
      "passed": true,
      "checks": {
        "expected_section_retrieved": true,
        "expected_section_cited": true,
        "answer_contains_expected_terms": true
      },
      "response": {
        "answer": "Employees must purchase economy airfare. Business-class airfare requires written approval from a vice president.",
        "citation": {
          "document": "Employee Expense Policy",
          "version": "2.0",
          "section": "3. Airfare"
        },
        "retrieved_chunks": [
          {
            "section": "3. Airfare",
            "distance": 0.31314281742910566
          },
          {
            "section": "4. Ground Transportation",
            "distance": 0.4643174856768598
          },
          {
            "section": "1. Meals",
            "distance": 0.4774149967780479
          }
        ]
      }
    },
    {
      "question": "My hotel costs $250. What do I need?",
      "passed": true,
      "checks": {
        "expected_section_retrieved": true,
        "expected_section_cited": true,
        "answer_contains_expected_terms": true
      },
      "response": {
        "answer": "Hotels are reimbursable up to $225 per night. A manager must approve higher rates before booking.",
        "citation": {
          "document": "Employee Expense Policy",
          "version": "2.0",
          "section": "2. Hotels"
        },
        "retrieved_chunks": [
          {
            "section": "2. Hotels",
            "distance": 0.3299007010222391
          },
          {
            "section": "5. Receipts",
            "distance": 0.368894834521612
          },
          {
            "section": "1. Meals",
            "distance": 0.4503105045845375
          }
        ]
      }
    },
    {
      "question": "Do I need a receipt for a $20 taxi?",
      "passed": true,
      "checks": {
        "expected_section_retrieved": true,
        "expected_section_cited": true,
        "answer_contains_expected_terms": true
      },
      "response": {
        "answer": "No. $20 is below the $25 threshold, so the stated requirement does not apply under this policy.",
        "citation": {
          "document": "Employee Expense Policy",
          "version": "2.0",
          "section": "5. Receipts"
        },
        "retrieved_chunks": [
          {
            "section": "5. Receipts",
            "distance": 0.3218649362156777
          },
          {
            "section": "6. Submission Deadline",
            "distance": 0.44206363244183144
          },
          {
            "section": "4. Ground Transportation",
            "distance": 0.4594500451459489
          }
        ]
      }
    },
    {
      "question": "Can I claim a limousine upgrade?",
      "passed": true,
      "checks": {
        "expected_section_retrieved": true,
        "expected_section_cited": true,
        "answer_contains_expected_terms": true
      },
      "response": {
        "answer": "Luxury vehicle upgrades are not reimbursable.",
        "citation": {
          "document": "Employee Expense Policy",
          "version": "2.0",
          "section": "4. Ground Transportation"
        },
        "retrieved_chunks": [
          {
            "section": "4. Ground Transportation",
            "distance": 0.3167883738983108
          },
          {
            "section": "2. Hotels",
            "distance": 0.500894293375563
          },
          {
            "section": "1. Meals",
            "distance": 0.5030920668357801
          }
        ]
      }
    },
    {
      "question": "Does the company reimburse gym memberships?",
      "passed": true,
      "checks": {
        "exact_refusal": true,
        "no_citation": true
      },
      "response": {
        "answer": "The provided policy does not answer this question.",
        "citation": null,
        "retrieved_chunks": [
          {
            "section": "4. Ground Transportation",
            "distance": 0.41995274425960316
          },
          {
            "section": "2. Hotels",
            "distance": 0.4273956845887261
          },
          {
            "section": "1. Meals",
            "distance": 0.4442459013608546
          }
        ]
      }
    }
  ]
}
```
