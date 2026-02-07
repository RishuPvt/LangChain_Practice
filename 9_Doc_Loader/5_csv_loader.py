from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='9_Doc_Loader/Social_Network_Ads.csv')

docs = loader.load()

print(len(docs))
print(docs[1])