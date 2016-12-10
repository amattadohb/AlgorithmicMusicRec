
import numpy as np

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

class PlaylistGeneration(Predictor):

	def __init__(self, distance, i, l):
		self.dtype = distance
		self.iterations = i 
		self.lambd = l 
		self.u = [] 
		self.r = []
		self.k = 0

		self.library = []
		#add all songs to the library
		for subpath, dirs, filelist in os.walk(os.path.join('./music', 'pickle')):
			for filename in filelist:
				if filename[-4:] == '.p':
					toAdd = pickle.load(open(os.path.join(os.path.join(subpath, filename) , "rb" )))
					self.library.append(toAdd)


	def train(self, feat1, feat2):
		if self.lambd == 0.0:
			self.calculate_lambda_bruh(vectors)
		self.u.append(self.get_dat_mean_yo(vectors))
		self.k = 1
		
		for _ in xrange(self.iterations):
			# E
			self.r = []
			for _ in xrange(self.k):
				self.r.append([])

			for song in self.library: #for song in library
				k, dist = self.closest_cluster(song, feat1, feat2)
				#print("lmbda = " + str(self.lambd) + ", distance = " + str(dist))

				if dist > self.lambd:
					#print("yo this went to " + str(self.k)) 
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

	def predict(self, name, author, layer, playlistlen, feat1, feat2):
		seed_title = name
		seed_artist = author
		
		#get song object of seed song
		seed = pickle.load(open(os.path.join('music', 'pickle', seed_artist + '_' + seed_title + '.p') , "rb" ))

		vector = np.concatenate((seed.getFeature(feat1), seed.getFeature(feat2)))
		k, _ = self.closest_cluster(vector)

		if layer == 0:
			length = 0
			for s in self.r[k]:
				if length >= playlistlen:
					break
				print(s.getTitle() + s.getArtist())
				length += 1



	def get_dat_mean_yo(self, feat1, feat2):
		mean = np.empty(len(self.library[0].getFeature(feat1)) + len(self.library[0].getFeature(feat2)))
		count = 0.0

		for song in self.library:
			mean = mean + np.concatenate((song.getFeature(feat1), song.getFeature(feat2)))
			count += 1.0
		return mean / count

	def calculate_lambda_bruh(self, feat1, feat2):
		mean = self.get_dat_mean_yo(feat1, feat2)
		s = 0
		count = 0.0
		

		if self.dtype == 'euc':
			for song in self.library:
				vector = np.concatenate((song.getFeature(feat1), song.getFeature(feat2)))
				count += 1.0
				s += np.linalg.norm(vector - mean) ** 2
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

		self.lambd = (s / count)

	def closest_cluster(self, song, feat1, feat2):
		vector = np.concatenate((song.getFeature(feat1), song.getFeature(feat2)))
		min_dist = float("inf")
		min_cluster = -1
		for k in xrange(self.k):

			mean = self.u[k]

			if self.dtype == 'euc':
				dist = np.linalg.norm(vector - mean) ** 2
			if self.dtype == 'max':
				max_dist = 0
				#find max distance
				for index in range(0, len(vector)):
					dist = abs(vector[index] - mean[index])
					if dist > max_dist:
						max_dist = dist
				dist = max_dist

			if dist < min_dist:
				min_dist = dist
				min_cluster = k
		return (min_cluster, min_dist)
			