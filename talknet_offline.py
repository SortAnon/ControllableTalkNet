if __name__ == '__main__':
    from controllable_talknet import *
    app.run_server(
        mode="external",
        debug=False,
        threaded=True,
    )
