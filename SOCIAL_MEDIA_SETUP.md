# Social Media Integrations

EmoTrack can now fetch recent posts or page through the full available account history from selected social platforms before running the mental-health analysis pipeline.

## Supported now

- Bluesky
  - Public read access for recent posts and paginated older history.
  - No token required for public accounts.
- Mastodon
  - Public read access for public statuses on open instances.
  - Use a full handle like `@user@mastodon.social` or provide the instance URL.
- X
  - Requires an API bearer token.
  - Set `X_BEARER_TOKEN` before starting Streamlit.

## Configure X

PowerShell:

```powershell
$env:X_BEARER_TOKEN="your-token-here"
.\.venv\Scripts\streamlit run app.py --server.port 8502
```

Or add it to Streamlit secrets:

```toml
X_BEARER_TOKEN = "your-token-here"
```

Place the secrets file at:

```text
.streamlit/secrets.toml
```

## Notes

- The app can fetch recent posts or continue paginating through older history.
- Full-history mode continues until the platform stops returning older posts, unless you set a safety cap.
- Private, protected, or unavailable accounts cannot be analyzed.
- Some platforms may rate-limit requests or require additional approval on the provider side.

## Test the connector directly

You can smoke-test the fetchers without opening Streamlit:

```powershell
.\.venv\Scripts\python test_social_media.py --platform bluesky --handle alice.bsky.social --limit 5
```

Bluesky full-history example:

```powershell
.\.venv\Scripts\python test_social_media.py --platform bluesky --handle alice.bsky.social --all-history --max-posts 1000
```

Mastodon example:

```powershell
.\.venv\Scripts\python test_social_media.py --platform mastodon --handle @user@mastodon.social --limit 5
```

X example:

```powershell
$env:X_BEARER_TOKEN="your-token-here"
.\.venv\Scripts\python test_social_media.py --platform x --handle XDevelopers --limit 5
```
