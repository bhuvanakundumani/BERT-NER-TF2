from bert import Ner

model = Ner("model_sep20/")

output = model.predict("Steve went to Paris")

print(output)