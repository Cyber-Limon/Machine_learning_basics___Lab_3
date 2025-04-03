from ucimlrepo import fetch_ucirepo
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MeanShift, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score
from prettytable import PrettyTable
import numpy as np
import time



# Импорт датасета
wholesale_customers = fetch_ucirepo(id=292)
x = wholesale_customers.data.features
y = wholesale_customers.data.targets



# Нормализация датасета
x_train = np.array(x)
x_scaled = preprocessing.StandardScaler().fit_transform(x_train)
y = np.array(y).ravel()



# Заполнение таблицы собственных векторов и значений
pca = PCA()
X = pca.fit_transform(x_scaled)
eigenvectors = pca.fit_transform(X)
eigenvalues = pca.explained_variance_



# Создание таблиц для представления результатов
table_kmeans = PrettyTable()
table_ms     = PrettyTable()
table_ac     = PrettyTable()
table_dbscan = PrettyTable()
table_best   = PrettyTable()

# Добавление колонок в таблицы
table_kmeans.field_names = ["Размерность", "ARI", "Точность", "Время выполнения", "n_clusters", "-"          ]
table_ms.field_names     = ["Размерность", "ARI", "Точность", "Время выполнения", "bandwidth" , "-"          ]
table_ac.field_names     = ["Размерность", "ARI", "Точность", "Время выполнения", "n_clusters", "-"          ]
table_dbscan.field_names = ["Размерность", "ARI", "Точность", "Время выполнения", "eps"       , 'min_samples']
table_best.field_names   = ["Размерность", "ARI", "Точность", "Время выполнения", "Параметр 1", "Параметр 2" ]

# Создание массивов для поиска лучших результатов
best_kmeans = [0, 0, 0, 0, "", ""]
best_ms     = [0, 0, 0, 0, "", ""]
best_ac     = [0, 0, 0, 0, "", ""]
best_dbscan = [0, 0, 0, 0, "", ""]



def search_for_best_results (table, best, dimension, ari, accuracy, period, parameter_1, parameter_2):
    if (best[1] < ari) or (best[1] == ari and best[3] > period):
        best[0] = dimension
        best[1] = ari
        best[2] = accuracy
        best[3] = period
        best[4] = parameter_1
        best[5] = parameter_2

        table.add_row([dimension, ari, accuracy, period, parameter_1, parameter_2])



def definition_of_accuracy(true, predict):
    count = 0

    for i in range(len(true)):
        if true[i] == predict[i]:
            count += 1

    return count / len(true)



def table_best_formation(table, best):
    table.add_row([best[0], best[1], best[2], best[3], best[4], best[5]])



# Цикл по всем размерностям
for component in range(1, len(eigenvalues) + 1):
    pca = PCA(n_components=component)
    X = pca.fit_transform(x_scaled)



    # Метод k-средних / kMeans
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k)

        start_time = time.time()
        kmeans.fit(X)
        end_time = time.time()

        ARI = adjusted_rand_score(y, kmeans.labels_)
        acc = definition_of_accuracy(y, kmeans.labels_)

        search_for_best_results(table_kmeans, best_kmeans, component, ARI, acc, end_time - start_time, k, "")



    # Метод сдвига среднего значения / Mean shift
    for b in range(2, 10):
        ms = MeanShift(bandwidth=b)

        start_time = time.time()
        ms.fit(X)
        end_time = time.time()

        ARI = adjusted_rand_score(y, ms.labels_)
        acc = definition_of_accuracy(y, ms.labels_)

        search_for_best_results(table_ms, best_ms, component, ARI, acc, end_time - start_time, b, "")



    # Метод агломеративной кластеризации / AgglomerativeClustering
    for k in range(2, 10):
        ac = AgglomerativeClustering(n_clusters=k)

        start_time = time.time()
        ac.fit(X)
        end_time = time.time()

        ARI = adjusted_rand_score(y, ac.labels_)
        acc = definition_of_accuracy(y, ac.labels_)

        search_for_best_results(table_ac, best_ac, component, ARI, acc, end_time - start_time, k, "")



    # Метод пространственной кластеризации для приложений с шумами / DBSCAN
    for min_samples in range(3, 10):
        for eps in range(1, 50):
            e = eps / 10
            dbscan = DBSCAN(eps=e, min_samples=min_samples)

            start_time = time.time()
            dbscan.fit(X)
            end_time = time.time()

            ARI = adjusted_rand_score(y, dbscan.labels_)
            acc = definition_of_accuracy(y, dbscan.labels_)

            search_for_best_results(table_dbscan, best_dbscan, component, ARI, acc, end_time - start_time, e, min_samples)



table_best_formation(table_best, best_kmeans)
table_best_formation(table_best,     best_ms)
table_best_formation(table_best,     best_ac)
table_best_formation(table_best, best_dbscan)



print("Метод k-средних / kMeans")
print(table_kmeans)

print("\nМетод сдвига среднего значения / Mean shift")
print(table_ms)

print("\nМетод агломеративной кластеризации / AgglomerativeClustering")
print(table_ac)

print("\nМетод пространственной кластеризации для приложений с шумами / DBSCAN")
print(table_dbscan)

print("\nЛучшие результаты")
print(table_best)
