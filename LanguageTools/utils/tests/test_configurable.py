def test_configurable():
    from LanguageTools.utils.configurable import Configurable

    class TestConfigurable0(Configurable):
        # noinspection PyUnusedLocal
        def __init__(self, arg1: str = "5", arg2: int = 5, arg3: int=1, arg4: str = None):
            ...

    class TestConfigurable1(Configurable):
        # noinspection PyUnusedLocal
        def __init__(self, arg1: str, arg2, *, arg3: int = 1, arg4: str = None):
            ...

    class TestConfigurable2(Configurable):
        # noinspection PyUnusedLocal
        def __init__(self, arg1: str, arg2, *, arg3, arg4: str = None):
            ...

    spec = TestConfigurable0.get_config_specification()
    assert spec == {
        "arg1": ("5", str),
        "arg2": (5, int),
        "arg3": (1, int),
        "arg4": (None, str)
    }

    spec = TestConfigurable1.get_config_specification()
    assert spec == {
        "arg3": (1, int),
        "arg4": (None, str)
    }

    try:
        # noinspection PyUnusedLocal
        spec = TestConfigurable2.get_config_specification()
        assert False, "Exception is not caught"
    except AssertionError:
        pass
