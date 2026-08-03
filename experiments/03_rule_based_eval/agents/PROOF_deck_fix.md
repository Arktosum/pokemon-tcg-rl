# PROOF: Deck Fix Applied

## dragapult_deck.csv (First 5 lines)
`
119
119
119
119
120
`

## iono_deck.csv (First 5 lines)
`
268
268
268
268
269
`

## dragapult.py (deck_path variable)
`python
deck_path = os.path.join(base_dir, "experiments", "03_rule_based_eval", "agents", "dragapult_deck.csv")
`

## iono.py (_deck_path variable)
`python
_deck_path = os.path.join(_agents_dir, "iono_deck.csv")
`
