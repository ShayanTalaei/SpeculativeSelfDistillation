"""Isolated vLLM process that serves the lockstep student/teacher pair.

The trainer (``ssd.paired_runner.PairedRunner``) spawns this module as a
subprocess and talks to it over a socket pair: INIT with the pair configuration,
then SYNC / GENERATE / RELEASE_TARGETS / CLOSE commands. Keeping vLLM in its own
process isolates its CUDA context and allocator from the TRL trainer.
"""

import argparse
import traceback
from multiprocessing.connection import Connection


def serve(conn):
    engine = None
    try:
        while True:
            command, payload = conn.recv()
            if command == "CLOSE":
                break
            if command == "INIT":
                from paired_vllm.lockstep_training import LockstepTrainingBridge

                engine = LockstepTrainingBridge(payload)
                conn.send((True, True))
                continue
            if engine is None:
                raise RuntimeError("INIT must precede every other command")
            conn.send((True, engine.call(command, payload)))
    except EOFError:
        pass
    except BaseException:  # noqa: BLE001 -- report subprocess failures to trainer
        conn.send((False, traceback.format_exc()))
    finally:
        if engine is not None:
            engine.close()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int, required=True)
    serve(Connection(parser.parse_args().fd))
