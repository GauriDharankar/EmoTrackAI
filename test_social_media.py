import argparse
import json
import os

from social_media import SocialFetchError, fetch_recent_posts


def build_parser():
    parser = argparse.ArgumentParser(description="Smoke test the EmoTrack social media connectors.")
    parser.add_argument("--platform", required=True, choices=["bluesky", "mastodon", "x"])
    parser.add_argument("--handle", required=True, help="Account handle to fetch.")
    parser.add_argument("--limit", type=int, default=5, help="Number of recent posts to fetch.")
    parser.add_argument(
        "--instance-url",
        default=None,
        help="Mastodon instance URL, for example https://mastodon.social",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print normalized post payloads as JSON.",
    )
    parser.add_argument(
        "--all-history",
        action="store_true",
        help="Paginate through the full available account history instead of only recent posts.",
    )
    parser.add_argument(
        "--max-posts",
        type=int,
        default=None,
        help="Optional safety cap when using --all-history.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    auth = {"x_bearer_token": os.getenv("X_BEARER_TOKEN")}

    try:
        posts = fetch_recent_posts(
            args.platform,
            args.handle,
            args.limit,
            instance_url=args.instance_url,
            auth=auth,
            fetch_all=args.all_history,
            max_posts=args.max_posts,
        )
    except SocialFetchError as exc:
        print(f"Fetch failed: {exc}")
        raise SystemExit(1) from exc

    print(f"Fetched {len(posts)} posts from {args.platform}.")

    if args.json:
        print(json.dumps([post.to_dict() for post in posts], indent=2))
        return

    for index, post in enumerate(posts, start=1):
        print()
        print(f"[{index}] {post.author} | {post.created_at or 'Unknown date'}")
        print(post.text)
        if post.url:
            print("URL:", post.url)


if __name__ == "__main__":
    main()
