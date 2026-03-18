import chromadb

# 1. 连接到你现有的本地数据库文件夹
db_path = "../../agent_final_db"
chroma_client = chromadb.PersistentClient(path=db_path)

# 2. 获取存储文本知识的集合
try:
    text_collection = chroma_client.get_collection(name="product_knowledge")
except Exception as e:
    print(f"找不到集合，请确认数据库路径是否正确: {e}")
    exit()

# 3. 获取集合中的所有数据
# 如果你的数据量很大，可以使用 text_collection.get(limit=10) 只看前10条
results = text_collection.get()

total_records = len(results['ids'])
print(f"📊 数据库中共有 {total_records} 条文本知识记录。\n" + "=" * 50)

# 4. 遍历并清晰地打印出每一条记录的内容
for i in range(total_records):
    doc_id = results['ids'][i]
    metadata = results['metadatas'][i]
    document = results['documents'][i]

    print(f"🆔 商品 ID: {doc_id}")
    print(f"🏷️ 元数据: {metadata}")
    print(f"📝 知识库实际存储的文本:")
    print(f"    {document}")
    print("-" * 50)