#!/usr/bin/env python3
"""Convert nested PersonalComment JSON into DiffuPercom split files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_json(path: Path) -> Dict[str, Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by article ID")
    return data


def convert(
    nested_data: Dict[str, Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    articles: Dict[str, Dict[str, Any]] = {}
    comments: List[Dict[str, Any]] = []

    for article_id, source_article in nested_data.items():
        if not isinstance(source_article, dict):
            raise ValueError(f"article {article_id!r} must be a JSON object")
        for field in ("title", "content"):
            if field not in source_article:
                raise ValueError(f"article {article_id!r} is missing {field!r}")

        article = {
            key: value for key, value in source_article.items() if key != "comments"
        }
        articles[str(article_id)] = article

        source_comments = source_article.get("comments", [])
        if not isinstance(source_comments, list):
            raise ValueError(f"article {article_id!r} has a non-list 'comments' value")

        for index, wrapped_comment in enumerate(source_comments):
            if not isinstance(wrapped_comment, dict) or len(wrapped_comment) != 1:
                raise ValueError(
                    f"comment {index} of article {article_id!r} must be a "
                    "single-key object"
                )
            record_id, source_comment = next(iter(wrapped_comment.items()))
            if not isinstance(source_comment, dict):
                raise ValueError(
                    f"comment record {record_id!r} of article {article_id!r} "
                    "must be a JSON object"
                )
            for field in ("content", "userinfo", "sentiment"):
                if field not in source_comment:
                    raise ValueError(
                        f"comment record {record_id!r} of article {article_id!r} "
                        f"is missing {field!r}"
                    )

            comment = dict(source_comment)
            comment["_id"] = str(article_id)
            comments.append(comment)

    return articles, comments


def write_json(path: Path, data: Any) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a nested PersonalComment JSON file into the articles.data "
            "and comments.data files consumed by DiffuPercom."
        )
    )
    parser.add_argument("input", type=Path, help="nested PersonalComment JSON file")
    parser.add_argument("output_dir", type=Path, help="destination split directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing articles.data and comments.data files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    articles_path = args.output_dir / "articles.data"
    comments_path = args.output_dir / "comments.data"

    existing = [path for path in (articles_path, comments_path) if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite {paths}; pass --overwrite")

    articles, comments = convert(read_json(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(articles_path, articles)
    write_json(comments_path, comments)
    print(
        f"Wrote {len(articles):,} articles and {len(comments):,} comments "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
