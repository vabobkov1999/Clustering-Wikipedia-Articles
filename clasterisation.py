import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram

# Массив названия статей из англоязычной википедии, распределен на три класса в соответсвии с вариантом
data = [
    #Mathematicians
    "Mathematician", "John Forbes Nash Jr. (mathematician)", "Mahāvīra (mathematician)", "Jim Simons (mathematician)",
    "Mathematics", "A Mathematician's Apology", "Mikhael Gromov (mathematician)", "Robert Morris (mathematician)",
    "John J. O'Connor (mathematician)", "I Am a Mathematician", "Robin Wilson (mathematician)",
    "Recreational mathematician", "Indian mathematician)", "Srinivasa Ramanujan", "John Dee (mathematician)",

    #History of Asia
    "History of Asia", "History of Asian art", "History of East Asia", "History of Southeast Asia",
    "Outline of South Asian history", "History of Asian Americans", "History of Central Asia",
    "Genetic history of East Asians", "Slavery in Asia", "Asian immigration to the United States",
    "Military history of Asia", "North Asia", "History of printing in East Asia", "History of the Middle East",
    "Christianity in Asia",

    #Media
    "Otitis media", "Social media", "Mass media", "Streaming media", "Media (communication)",
    "Vice Media", "Media studies", "Influence of mass media", "Alternative media", "Media conglomerate",
    "Virgin Media", "Multi-media", "Digital media", "Gawker Media", "Media ethics"
]
print(data) # проверка правильности того, что статьи забиты верно(сделал для себя)

text_from_wiki = [] # создаем пустой массив

for x in data:
    text_from_wiki.append(wikipedia.page(title=x).content) # забивает массив text_from_wiki содержимым всех статей, которые были записаны раньше

model = TfidfVectorizer(stop_words={'english'})# создаем модель TfidVectorizer со стопсловом english, масиив статей состоящих из большого числа слов, мы переводим их в некоторые коэффициенты для выявления фитч
X = model.fit_transform(text_from_wiki)# забиваем в модель все значения fit_transform(преобразует в числовые данные). В результате получаем вектор Х в котором находится массив статей и который является вектором фитч

#START Elbow метод проверки количества кластеров
clusters = range(2, 8) # массив состоящий из чисел от 2 до 8
tmp = []
for x in clusters: # берём число от 2 до 8 по порядку, создаем модель KMeans, с помощью неё мы находим весовой центр и этот метод передвигает в лучшую позицию, с каждой итерацией добавляя какие-то нужные данные и убирая лишние. max_iter - число итераций которое он будет производить
    model = KMeans(n_clusters=x, init='k-means++', n_init=10, max_iter=200)# c помощью её строим Elbow кривую
    Y = model.fit(X) # запоминаем все полученные значения
    tmp.append(Y.inertia_)

plt.plot(clusters, tmp, 'go-') # строим график
plt.xlabel("Number of clusters") # подпись на оси х
plt.ylabel("Squared distances") # подпись на оси у
plt.title("Elbow method") # название
plt.show() # показать график
#END Elbow метод проверки количества кластеров

#START KMeans
print ('\n', "KMeans method", '\n')
clusters = 3 # число кластеров
model = KMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=200) # создаем точно такую же модель KMeans, как в предыдущем методе
model.fit(X) # забиваем данные
labels = model.labels_ # дастает из модели название кластеров
clusters_kmeans = pd.DataFrame(list(zip(data, labels)), columns=['title', 'cluster']) # модель метода DataFrame
print(clusters_kmeans) # вывод
#END KMeans

#START MiniBatchKMeans, переводит не все центры, а только часть из них
print ('\n', "MiniBatchKMeans method", '\n')
clusters = 3
model = MiniBatchKMeans(n_clusters=clusters, init='k-means++', n_init=10, max_iter=200)
model.fit(X)
labels = model.labels_
clusters_minibatch = pd.DataFrame(list(zip(data, labels)), columns=['title', 'cluster'])
print(clusters_minibatch)
#END MiniBatchKMeans

#START DBSCAN метод, хороший для сгруппированного числа элементов, не нужно знать число кластеров, смотрит на данные
print ('\n', "DBSCAN method", '\n')
model = DBSCAN(eps=0.9, min_samples=5, metric='euclidean', algorithm='brute') # eps- расстояние между объектами, min_samples- минимальное число элементов в кластере, metric- евклидова более стабильная
model.fit(X)
labels = model.labels_
clusters_dbscan = pd.DataFrame(list(zip(data, labels)), columns=['title', 'cluster'])
print(clusters_dbscan)
#END DBSCAN

#START hierarchy - дерево, берет на первом этапе каждую пару и смотрит, какая  из них ближе всего находится и запихивает в один кластер, потом пытается совместить пары кластеров, возможно что ни одна пара не совместится, тогда в качестве класса оставляет один элемент. Метод объединяет все элементы пока недоходит до одного класса в репозитории есть картинка с этим методом
samples = X.toarray()
mergings = linkage(samples, method='ward')
dendrogram(mergings, labels=data, leaf_font_size=5, orientation='right')
plt.show()
#END hierarchy
