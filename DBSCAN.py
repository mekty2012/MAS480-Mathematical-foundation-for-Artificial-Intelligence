import random
from math import pi, sin, cos
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from datetime import datetime


NOISE = -2
UNDEFINED = -1

def normalize(data, mean, variance):
    # data : N * d
    # mean : d
    # variance : d
    result = []
    mean_list = [0 for i in range(len(mean))]
    variance_list = [0 for i in range(len(variance))]



def euclidean_distance(first, second, dim):
    first = first[0]
    second = second[0]
    if len(first) != len(second):
        raise ValueError("Dimension does not fit")
    result = 0.0
    for i in range(len(first)):
        result += (first[i] - second[i]) ** dim

    return np.power(result, 1/dim)

def manhattan_distance(first, second):
    first = first[0]
    second = second[0]
    if len(first) != len(second):
        raise ValueError("Dimension does not fit")
    result = 0.0
    for i in range(len(first)):
        result += abs(first[i] - second[i])

    return result

def pearson_correlation_distance(first, second):
    first = first[0]
    second = second[0]
    if len(first) != len(second):
        raise ValueError("Dimension does not fit")
    first_ave = sum(first) / len(first)
    second_ave = sum(second) / len(second)
    xy_cor = 0.0
    xx_cor = 0.0
    yy_cor = 0.0
    for i in range(len(first)):
        xy_cor += (first[i] - first_ave) * (second[i] - second_ave)
        xx_cor += (first[i] - first_ave) ** 2
        yy_cor += (second[i] - second_ave) ** 2
    rho = xy_cor / np.sqrt(xx_cor * yy_cor)
    return 1 - rho

def spearman_rank_correlation_distance(first, second):
    first = first[0]
    second = second[0]
    if len(first) != len(second):
        raise ValueError("Dimension does not fit")
    first_with_index = [(i, first[i]) for i in range(len(first))].sort(key=lambda x : x[1])
    second_with_index = [(i, second[i]) for i in range(len(first))].sort(key=lambda x: x[1])
    result = 0.0
    for i in range(len(first)):
        result += (first[i] - second[i]) ** 2
    return result * 6 / (len(first) * (len(first) ** 2 - 1))

def DBSCAN(points, dist_func, eps, min_pts, clusters):
  cluster_counter = 0
  for point in points:
    if point[1] != UNDEFINED: continue
    neighbors = range_query(points, dist_func, point, eps)
    if len(neighbors) < min_pts:
      point[1] = NOISE
      continue
    cluster_counter += 1
    clusters.append((list(), list(), list()))

    point[1] = cluster_counter
    clusters[cluster_counter - 1][0].append(point[0][0])
    clusters[cluster_counter - 1][1].append(point[0][1])
    clusters[cluster_counter - 1][2].append(point[0][2])


    seed_set = neighbors.copy()
    seed_set.remove(point)

    while len(seed_set) != 0:
      neighbor = seed_set.pop(0)
      if neighbor[1] == NOISE:
        neighbor[1] = cluster_counter
      if neighbor[1] != UNDEFINED:
        continue
      neighbor[1] = cluster_counter
      clusters[cluster_counter - 1][0].append(neighbor[0][0])
      clusters[cluster_counter - 1][1].append(neighbor[0][1])
      clusters[cluster_counter - 1][2].append(neighbor[0][2])
      other_neighbors = range_query(points, dist_func, neighbor, eps)
      if len(other_neighbors) >= min_pts:
        seed_set.extend(other_neighbors)

def clean(points):
  for point in points:
    point[1] = UNDEFINED

def range_query(points, dist_func, point, eps):
  result = list()
  for other in points:
    if dist_func(point, other) <= eps:
      result.append(other)
  return result

if __name__ == "__main__":
  point_list = list()
  coordinate_list = (list(), list(), list())
  # eps : 1/3 , minpts : 40
  for _ in range(500):
    angle = np.random.uniform(-np.pi, np.pi)
    cos = 6 * np.cos(angle)
    sin = 6 * np.sin(angle)
    x = cos + np.random.normal(0, 0.5)
    y = 0 + np.random.normal(0, 0.5)
    z = sin + 3 + np.random.normal(0, 0.5)
    point_list.append([(x, y, z), UNDEFINED])
    coordinate_list[0].append(x)
    coordinate_list[1].append(y)
    coordinate_list[2].append(z)

  for _ in range(500):
    angle = np.random.uniform(-np.pi, np.pi)
    cos = 6 * np.cos(angle)
    sin = 6 * np.sin(angle)
    x = 0 + np.random.normal(0, 0.5)
    y = cos + np.random.normal(0, 0.5)
    z = sin - 3 + np.random.normal(0, 0.5)
    point_list.append([(x, y, z), UNDEFINED])
    coordinate_list[0].append(x)
    coordinate_list[1].append(y)
    coordinate_list[2].append(z)

  plt.figure(1)
  axes = plt.axes(projection='3d')
  axes.scatter3D(coordinate_list[0], coordinate_list[1], coordinate_list[2])
  plt.show()

  colors = {0:'g', 1:'r', 2:'c', 3:'m', 4:'y'}

  bef = datetime.today()
  euclidean_clusters = list()
  DBSCAN(point_list, lambda x, y:euclidean_distance(x, y, 2), 1, 7, euclidean_clusters)
  print("Euclidean : %ds"%(datetime.today()-bef).total_seconds())
  plt.figure(2)
  axes = plt.axes(projection='3d')

  for i in range(len(euclidean_clusters)):
    axes.scatter3D(euclidean_clusters[i][0], euclidean_clusters[i][1], euclidean_clusters[i][2], c = colors.get(i, 'b'))
  plt.show()
  clean(point_list)

  bef = datetime.today()
  manhattan_clusters = list()
  DBSCAN(point_list, manhattan_distance, 1.5, 7, manhattan_clusters)
  print("Manhattan : %ds"%(datetime.today()-bef).total_seconds())
  plt.figure(3)
  axes = plt.axes(projection='3d')

  for i in range(len(manhattan_clusters)):
    axes.scatter3D(manhattan_clusters[i][0], manhattan_clusters[i][1], manhattan_clusters[i][2], c = colors.get(i, 'b'))
  plt.show()
  clean(point_list)

  bef = datetime.today()
  pearson_clusters = list()
  DBSCAN(point_list, pearson_correlation_distance, .002, 10, pearson_clusters)
  print("Pearson : %ds"%(datetime.today()-bef).total_seconds())
  plt.figure(4)
  axes = plt.axes(projection='3d')

  for i in range(len(pearson_clusters)):
    axes.scatter3D(pearson_clusters[i][0], pearson_clusters[i][1], pearson_clusters[i][2], c = colors.get(i, 'b'))
  plt.show()
  clean(point_list)

  bef = datetime.today()
  spearman_clusters = list()
  DBSCAN(point_list, spearman_rank_correlation_distance, .3, 10, spearman_clusters)
  print("Spearman : %ds"%(datetime.today()-bef).total_seconds())
  plt.figure(5)
  axes = plt.axes(projection='3d')

  for i in range(len(spearman_clusters)):
    axes.scatter3D(spearman_clusters[i][0], spearman_clusters[i][1], spearman_clusters[i][2], c = colors.get(i, 'b'))
  plt.show()
