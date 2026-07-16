# Data schema

This repository contains two representations of PersonalComment data:

- `data_demo/data_demo.json` uses the original nested representation: each
  article owns a `comments` list, and every list item wraps one comment record.
- DiffuPercom uses the split representation described below, where articles
  and comments are stored separately and each comment's `_id` points to its
  article.

Convert a nested file to the split representation with
`scripts/prepare_personalcomment.py`.

The PersonalComment dataset was collected from Sina News between November
2021 and August 2023 and includes article text, comments, age, gender,
location, profile description, and an automatically assigned sentiment
polarity.

Reported split statistics:

| Item | Total | Train | Validation | Test |
|---|---:|---:|---:|---:|
| Articles | 67,597 | 65,597 | 1,000 | 1,000 |
| Comments | 580,748 | 564,191 | 8,833 | 7,724 |
| Personalized attributes | 580,748 | 564,191 | 8,833 | 7,724 |

Every model-ready split directory contains two UTF-8 JSON files.

`articles.data` is an object keyed by article ID:

```json
{
  "example-1": {
    "title": "Article title",
    "content": ["First paragraph.", "Second paragraph."]
  }
}
```

`comments.data` is a list. Each comment's `_id` must match a key in
`articles.data`:

```json
[
  {
    "_id": "example-1",
    "content": "Comment text",
    "sentiment": "正面",
    "userinfo": {
      "gender": "f",
      "age": 30,
      "location": "其他",
      "description": "Synthetic profile"
    }
  }
]
```

Accepted sentiment labels are `负面`, `正面`, and `中性`. Gender is encoded as
`f` or `m`. The files under `data/example/` are synthetic and are intended only
to document the parser contract.

The paper describes filtering duplicate or irrelevant material, retaining
article lengths from 200 to 2,000 words and comment lengths from 5 to 128,
removing records with incomplete persona attributes, filtering sensitive
terms, and replacing recognized email addresses, phone numbers, and account
identifiers with generic placeholders. Users remain responsible for following
the dataset's license, access terms, and privacy requirements.
