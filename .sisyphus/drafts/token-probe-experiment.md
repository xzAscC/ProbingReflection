# Draft: Token Probe Experiment

## Requirements (confirmed)
- Create a new git branch for experiments
- Two Python files for two tasks:
  1. Find top 50 pos/neg tokens per concept using discriminative method, decode and display
  2. Train linear probe on tokens closest to pos/neg centers, test separability across 10 layers

## Open Questions
- **Model**: Which model to probe?
- **Concepts**: What specific concepts to analyze?
- **Data source**: Where does the token/concept data come from?
- **Discriminative method**: Which specific method? (mass-mean, contrast-cons, etc.)
- **Distance metric**: How to measure "closest to center"? (cosine, euclidean)
- **Token selection criteria**: Top-K based on what score?

## Technical Decisions
- (pending user input)

## Scope Boundaries
- INCLUDE: Two Python scripts, new branch
- EXCLUDE: (pending)
