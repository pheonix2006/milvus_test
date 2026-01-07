from pymilvus import connections, db

def connect_milvus():
    """初始化 Milvus 连接演示"""
    print("正在连接到 Milvus...")
    # connections.connect(host='localhost', port='19530')
    return True

def main():
    print("Hello from milvus-test!")
    connect_milvus()
    print("项目初始化成功！")


if __name__ == "__main__":
    main()
