if __name__ == "__main__":
    import os
    from controllable_talknet import *

    if os.path.exists("/talknet/is_docker"):
        app.run_server(
            host="0.0.0.0",
            mode="external",
            debug=False,
            threaded=True,
        )
    else:
        app.run_server(
            mode="external",
            debug=False,
            threaded=True,
        )
