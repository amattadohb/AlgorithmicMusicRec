from playlistgen import *
from playlistgeneration import *

# generator = PlaylistGen('euc', 10, 0.0, 15, 3)
# songs = generator.suggest('Meme Generator', 'Dan Deacon ', 'sr', 'zcr', '', '')

generator = PlaylistGeneration('euc', 10, 0.0)
generator.train('sr', 'zcr')
generator.predict('Meme Generator', 'Dan Deacon ', 0, 15, 'sr', 'zcr')