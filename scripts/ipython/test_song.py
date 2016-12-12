from songsuggestion import *

distances = ['euc', 'max']
num_suggestions = 15
songs = [('Touch the Sky (Featuring Lupe Fiasco)', 'Kanye West')]
features = ['sr', 'sc', 'rms', 'zcr', 'mfcc']

'''
for dist in distances:
	for title, artist in songs:
		for feat1 in features:
			for feat2 in features:
				if feat2 == feat1:
					continue
				print '\n\n\n\n', title, artist, feat1, feat2, dist
				test = SongSuggestion(dist, num_suggestions)
				test.suggest(title, artist, feat1, feat2)
'''
title = "Mind Mischief"
artist = "Tame Impala"
feat1 = 'mfcc'
feat2 = 'sc'
dist = 'euc'
print '\n\n\n\n', title, artist, feat1, feat2, dist
test = SongSuggestion(dist, num_suggestions)
test.suggest(title, artist, feat1, feat2)

title = "Mind Mischief"
artist = "Tame Impala"
feat1 = 'mfcc'
feat2 = 'rms'
dist = 'max'
print '\n\n\n\n', title, artist, feat1, feat2, dist
test = SongSuggestion(dist, num_suggestions)
test.suggest(title, artist, feat1, feat2)

#test = SongSuggestion("euc", 45)
#test.suggest("Broken Jaw", "Foster the People", "sr", "zcr")