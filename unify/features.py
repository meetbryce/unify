# As we add features that Connectors may depend on we should add them
# here with a key. Then a Connector should use the 'requires' stanza
# to declare that it needs a feature. We use this more explainable and
# granular approach rather than just versioning the whole interpreter.

class Features:
    FEATURES = [
        'auth.defaults',
        'connector.oauth'
    ]

    @staticmethod
    def has_feature(feature: str):
        return feature in Features.FEATURES
