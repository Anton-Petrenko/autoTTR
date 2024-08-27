# Testing

import internal.types as types
import internal.data as data

# Types
colors: list = data.listColors()

# Deck
deck: types.Deck = types.Deck(colors)
assert list(deck.cards) == colors
assert deck.count == len(colors)
