import stanza
from stanza import DownloadMethod


class ClinicalData:
    def __init__(self):
        super().__init__()
        self.model = None
        try:
            self.model = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'},download_method=DownloadMethod.REUSE_RESOURCES)
        except:
            stanza.download('en', package='mimic', processors={'ner': 'i2b2'})
            self.model = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'})

    def classify(self,notes):
        doc = self.model(notes)
        return doc