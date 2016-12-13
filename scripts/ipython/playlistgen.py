import numpy as np
import os 
import pickle
from songsuggestion import *
from random import shuffle

class Song:

	def __init__(self, name, artiste, feat):
		self.title = name
		self.artist = artiste
		self.features = feat

	def getTitle(self):
		return self.title

	def getArtist(self):
		return self.artist

	def getFeature(self, name):
		return self.features[name]

	def addFeature(self, dictl):
		self.features = dictl

class PlaylistGen:
	def __init__(self, distance, i, l, length, layer0):
		self.dtype = distance
		self.iterations = i 
		self.lambd = l 
		self.plength = length
		self.layer0factor = layer0
		self.u = [] 
		self.r = []
		self.k = 0
		self.library = []
		self.cluster_heap = []
		self.playlist = []

	def suggest(self, name, author, feat1, feat2, feat3, feat4, cluster_range):
		firstlayer = SongSuggestion(self.dtype, self.plength*self.layer0factor)
		layer0 = firstlayer.suggest(name, author, feat1, feat2, False)

		#add songs from layer0 to library
		for x in layer0:
			self.library.append(x[1])

		#layer 1
		self.train(feat3, feat4)
		self.predict(name, author, feat3, feat4, cluster_range)

		#print out playlist
		shuffle(self.playlist)
		print("Playlist generated for " + name + " by " + author)
		for song in self.playlist:
			print song.getTitle(), song.getArtist()
		return self.playlist


	#training for lambda means clustering for layer 1
	def train(self, feat1, feat2):
		if self.lambd == 0.0:
			self.calculate_lambda(feat1, feat2)
		self.u.append(self.calculate_mean(feat1, feat2))
		self.k = 1
		
		for ite in xrange(self.iterations):
			# E
			self.r = []
			for _ in xrange(self.k):
				self.r.append([])

			for song in self.library: #for song in library
				k, dist = self.closest_cluster(song, feat1, feat2)
				# print("lmbda = " + str(self.lambd) + ", distance = " + str(dist))

				if dist > self.lambd:
					self.k += 1
					self.r.append([])
					self.r[self.k - 1].append(song)
					self.u.append(np.concatenate((song.getFeature(feat1), song.getFeature(feat2))))
				else:
					self.r[k].append(song)

			# M
			self.u = []
			for k in xrange(self.k):
				#print("Cluster " + str(k) + ":")
				mean = np.zeros(len(self.library[0].getFeature(feat1)) + len(self.library[0].getFeature(feat2)))
				count = 0.0
				for song in self.r[k]:
					vector = np.concatenate((song.getFeature(feat1), song.getFeature(feat2)))
					mean = mean + vector
					count += 1.0

				if count == 0:
					self.u.append(np.zeros(len(self.library[0].getFeature(feat1)) + len(self.library[0].getFeature(feat2))))
				else:
					self.u.append(mean / count)

	#predicting for lambda means clustering for layer 1
	def predict(self, name, author, feat1, feat2, cluster_range=0):
		seed_title = name
		seed_artist = author
		
		#get song object of seed song
		seed = pickle.load(open(os.path.join('music', 'pickle', seed_artist + '_' + seed_title + '.p') , "rb" ))

		#shuffle song order in clusters
		for k in xrange(self.k):
			shuffle(self.r[k])

		###########we don't use this yet tho #################
		self.cluster_ranking(seed, feat1, feat2)
		
		counter = 0
		song_index = 0
		length = 0
		if cluster_range == 0:
			while(length < self.plength):
				c_index = counter % self.k
				cluster = self.r[c_index]

				#add a song
				if song_index < len(cluster):
					self.playlist.append(cluster[song_index])
					length += 1

				counter += 1

				if counter % self.k == 0:
					song_index += 1
		else:
			top_clusters = nsmallest(int(cluster_range), self.cluster_heap)
			while self.get_num_songs(top_clusters) < self.plength:
				cluster_range += 1
				top_clusters = nsmallest(int(cluster_range), self.cluster_heap)

			while(length < self.plength):
				c_index = counter % cluster_range
				cluster = top_clusters[c_index][1]

				#add a song
				if song_index < len(cluster):
					self.playlist.append(cluster[song_index])
					length += 1

				counter += 1

				if counter % cluster_range == 0:
					song_index += 1




		#add seed song to playlist if not added
		if seed not in self.playlist:
			del self.playlist[0]
			self.playlist.append(seed)

		print "Cluster Range:", cluster_range



	def get_num_songs(self, top_clusters):
		count = 0
		for cluster in top_clusters:
			count += len(cluster[1])
		return count


	def calculate_mean(self, feat1, feat2):
		mean = np.zeros(len(self.library[0].getFeature(feat1)) + len(self.library[0].getFeature(feat2)))
		count = 0.0

		for song in self.library:
			mean = mean + np.concatenate((song.getFeature(feat1), song.getFeature(feat2)))
			count += 1.0
		return mean / count

	def calculate_lambda(self, feat1, feat2):
		mean = self.calculate_mean(feat1, feat2)
		s = 0.0
		count = 0.0
		

		if self.dtype == 'euc':
			for song in self.library:
				vector = np.concatenate((song.getFeature(feat1), song.getFeature(feat2)))
				count += 1.0
				s += distance.euclidean(vector, mean)# ** 2
		if self.dtype == 'max':
			for song in self.library:
				vector = np.concatenate((song.getFeature(feat1), song.getFeature(feat2)))
				count += 1.0

				max_dist = 0
				#find max distance
				for index in range(0, len(vector)):
					dist = abs(vector[index] - mean[index])
					if dist > max_dist:
						max_dist = dist
				s += max_dist

		self.lambd = (s / (count))

	def closest_cluster(self, song, feat1, feat2):
		vector = np.concatenate((song.getFeature(feat1), song.getFeature(feat2)))
		min_dist = 0
		min_cluster = -1
		for k in xrange(self.k):
			mean = self.u[k]

			if self.dtype == 'euc':
				dist = distance.euclidean(vector, mean)# ** 2
			if self.dtype == 'max':
				max_dist = 0
				#find max distance
				for index in range(0, len(vector)):
					dist = abs(vector[index] - mean[index])
					if dist > max_dist:
						max_dist = dist
				dist = max_dist

			if dist < min_dist or min_cluster == -1:
				min_dist = dist
				min_cluster = k
		return (min_cluster, min_dist)
			
	def cluster_ranking(self, song, feat1, feat2):
		vector = np.concatenate((song.getFeature(feat1), song.getFeature(feat2)))

		for k in xrange(self.k):
			mean = self.u[k]

			if self.dtype == 'euc':
				dist = distance.euclidean(vector, mean)
			if self.dtype == 'max':
				max_dist = 0
				#find max distance
				for index in range(0, len(vector)):
					dist = abs(vector[index] - mean[index])
					if dist > max_dist:
						max_dist = dist
				dist = max_dist

			heappush(self.cluster_heap, (dist, self.r[k]))


