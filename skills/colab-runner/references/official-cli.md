# Official Colab CLI reference

Colab Runner targets Google's `google-colab-cli` package and shells out to its
documented `colab` command. It does not import the package's private modules.

Official sources:

- Repository: https://github.com/googlecolab/google-colab-cli
- Package: https://pypi.org/project/google-colab-cli/
- Announcement: https://developers.googleblog.com/introducing-the-google-colab-cli/

## Authentication recovery

Prefer Application Default Credentials for non-interactive agents. Run this in
a user-controlled terminal:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

Then verify without allocating compute:

```bash
colab --auth=adc version
colab --auth=adc whoami
colab --auth=adc sessions
```

The global `--auth=adc` option must precede the command. Bare commands can enter
the CLI's interactive OAuth flow, which is intentionally unavailable inside the
stdio MCP server.

When `gcloud` is unavailable, complete the official CLI's remote copy/paste
OAuth2 flow once in a user-controlled terminal:

```bash
colab --auth=oauth2 whoami
export COLAB_RUNNER_AUTH=oauth2
colab --auth=oauth2 sessions
```

The MCP server reads `COLAB_RUNNER_AUTH` at startup and accepts `adc` or
`oauth2`; it defaults to `adc`. The cached OAuth2 token must exist before Codex
starts because MCP subprocesses receive no interactive stdin.

The pinned CLI keeps `whoami` out of its top-level command index but documents
it through `colab help whoami` and its bundled agent skill. Colab Runner uses it
only for the doctor's read-only scope check and never returns the account email.

Do not confuse CLI authentication with `colab auth`. The latter injects GCP
credentials into an already-running VM and is interactive.

## Command boundary

Colab Runner composes these official commands:

```text
colab --auth=adc version
colab --auth=adc whoami
colab --auth=adc sessions
colab --auth=adc status -s NAME
colab --auth=adc new -s NAME [--gpu GPU | --tpu TPU]
colab --auth=adc install -s NAME [-r FILE | PACKAGE ...]
colab --auth=adc exec -s NAME --timeout SECONDS -f FILE
colab --auth=adc download -s NAME /content/REMOTE LOCAL
colab --auth=adc log -s NAME -o RUN.ipynb
colab --auth=adc stop -s NAME
```

Supported accelerator values are T4, L4, G4, H100, A100, TPU v5e1, and TPU
v6e1. Availability depends on the account and current Colab capacity.

## Recovery

- Authentication failure: refresh ADC with all four scopes above.
- Accelerator allocation failure: do not silently select a different device;
  report it and ask whether CPU or another accelerator is acceptable.
- Execution failure: retrieve only outputs known to be complete; do not replay
  automatically.
- Cleanup failure: call `colab_sessions`, inspect the generated session, and
  use `colab_stop_session` when it is still assigned.
