from milvus_test.main import connect_milvus

def test_connect_milvus():
    assert connect_milvus() is True

def test_placeholder():
    assert 1 + 1 == 2
