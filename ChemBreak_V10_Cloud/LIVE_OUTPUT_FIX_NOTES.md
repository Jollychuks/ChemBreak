# V10 live-output controller fix

The V10 pipeline already contained live progress, ETA, and 20-second heartbeat messages.

The issue was the notebook controller. It still launched the child pipeline with a plain `subprocess.run(...)` call. In some Colab sessions, child-process output was not being surfaced continuously in the cell.

This release changes the controller notebook to:

- use `subprocess.Popen(...)`
- merge stderr into stdout
- stream each output line into Colab immediately
- force `PYTHONUNBUFFERED=1`
- keep Python `-u`
- print V10-specific stage banners
- verify after the GitHub clone that the loaded pipeline really contains the live-progress build

No ChemBreak benchmark logic was changed by this fix.
