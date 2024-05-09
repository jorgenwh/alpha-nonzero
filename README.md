# Alpha-NonZero

<p align="center"><img src="assets/aichess.png" width="50%"></p>

## Development plan

### Experimental (Python)
- [x] Obtain chess games dataset (Lichess?)
- [x] Implement PGN parser
- [x] Implement board representation and output similar to GLCWS
- [ ] Implement transformer architecture used in GLCWS
- [ ] Implement Stockfish evaluation for chess positions to generate training data
- [ ] Set up training pipeline where we train the model on auto-generated training data and evaluate that it improves in strength

### Implement search mechanism (C++, CUDA, cuDNN)
* [ ] ...