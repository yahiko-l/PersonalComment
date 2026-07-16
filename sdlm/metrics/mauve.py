from evaluate import load
mauve = load('mauve')
predictions = ["hello world", "goodnight moon"]
references = ["hello world",  "goodnight moon"]
mauve_results = mauve.compute(predictions=predictions, references=references)
print(mauve_results)


# from evaluate import load
# mauve = load('mauve')
# predictions = ["hello world", "goodnight moon"]
# references = ["hello world",  "goodnight moon"]
# mauve_results = mauve.compute(predictions=predictions, references=references)
# print(mauve_results.mauve)


# from evaluate import load
# mauve = load('mauve')
# predictions = ["hello world", "goodnight moon"]
# references = ["hello there", "general kenobi"]
# mauve_results = mauve.compute(predictions=predictions, references=references)
# print(mauve_results.mauve)
# 0.27811372536724027
