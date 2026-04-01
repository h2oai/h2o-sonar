# Copyright 2018-2026 H2O.ai, Inc. All rights reserved.


class PersistedData:
    def __init__(self, data_location, upload_config=None):
        """This class represents data stored either on local disk, in the cloud or
        any other distributed store.

        Parameters
        ----------
        data_location : str
            String path to the data. Currently only local
            path is supported.
        upload_config : dict
            configuration used during data fetching. Currently
            supported values depends on the data backend chosen.

            H2O backend: all parameters accepted by h2o.import_file()

            Pandas backend: not yet supported

        """
        if not upload_config:
            upload_config = {}
        self.upload_config = upload_config
        self.data_location = data_location
