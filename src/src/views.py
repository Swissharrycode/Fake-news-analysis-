from multiprocessing import context
from django.shortcuts import render
from django.views.generic import View
from django.contrib import messages

# ml lib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

class index(View):
    def get(self, request):
        context = {}
        context['news'] = ''
        return render(request, 'index.html', context)

    def post(self, request):
        # data
        data = pd.read_csv("D:\\workspaces\\fakeNewsDetection\\news.csv")
        # split
        x = np.array(data["title"])
        y = np.array(data["label"])

        # transform
        cv = CountVectorizer()
        x = cv.fit_transform(x)
        xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
        model = MultinomialNB()
        # train
        model.fit(xtrain, ytrain)
        text = request.POST.get('news')
        data = cv.transform([text]).toarray()
        predicetion = model.predict(data)
        context = {"news": text}
        if predicetion[0] == 'REAL':
            messages.success(request, 'News is Real')
        else:
            messages.error(request, 'News is fake')
        return render(request, 'index.html', context)