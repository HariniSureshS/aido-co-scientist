# Project Structure

```
aido-co-scientist/
│
├── README.md                          # Quick start, overview, how it works
├── GUIDE.md                           # Docker, GPU/Colab, CLI reference, testing
├── ARCHITECTURE.md                    # Detailed system design (80KB)
├── research_paper.md                  # Research paper draft with results & analysis
├── AI Scientist Homework (1).pdf      # Original assignment spec
│
├── co_scientist/                      # Main package (all source code)
│   ├── cli.py                         # CLI entry point (co-scientist run/batch)
│   ├── config.py                      # RunConfig, defaults
│   ├── defaults.py                    # Default config loader
│   ├── defaults.yaml                  # Predefined model configs & HP search spaces
│   │
│   ├── agents/                        # Multi-agent system
│   │   ├── coordinator.py             # Orchestrates pipeline, resolves conflicts
│   │   ├── ml_engineer.py             # Model selection & HP strategy
│   │   ├── data_analyst.py            # Data profiling & preprocessing decisions
│   │   ├── biology.py                 # Domain expertise & biological interpretation
│   │   ├── research.py                # Literature search (Semantic Scholar, PubMed)
│   │   ├── react.py                   # Thought → Action → Observation agent loop
│   │   ├── tree_search.py             # MCTS-inspired branching for model exploration
│   │   ├── debate.py                  # Agent debate before major decisions
│   │   ├── tools.py                   # Tool registry for agent actions
│   │   ├── interactive.py             # Interactive mode user prompts
│   │   ├── base.py                    # BaseAgent class
│   │   └── types.py                   # AgentMessage, Decision types
│   │
│   ├── data/                          # Data loading & preprocessing
│   │   ├── loader.py                  # HuggingFace dataset loading
│   │   ├── profile.py                 # Modality/task detection, statistics
│   │   ├── preprocess.py              # k-mers, log1p+HVG, scaling per modality
│   │   ├── split.py                   # Train/val/test splitting (stratified)
│   │   └── types.py                   # LoadedDataset, DatasetProfile
│   │
│   ├── modeling/                      # Model implementations
│   │   ├── registry.py                # Model builders for all tiers
│   │   ├── trainer.py                 # Training orchestration
│   │   ├── mlp.py                     # MLP classifier/regressor (PyTorch)
│   │   ├── ft_transformer.py          # Feature Tokenizer + Transformer
│   │   ├── bio_cnn.py                 # Multi-scale 1D CNN for biological sequences
│   │   ├── ensemble.py                # Stacking ensemble (Ridge/Logistic meta-learner)
│   │   ├── foundation.py              # AIDO GPU embedding extraction
│   │   ├── aido_finetune.py           # End-to-end AIDO backbone fine-tuning
│   │   ├── custom_model.py            # LLM-designed custom architectures
│   │   ├── hp_search.py               # Optuna Bayesian hyperparameter optimization
│   │   └── types.py                   # ModelConfig, TrainedModel
│   │
│   ├── evaluation/                    # Metrics & ranking
│   │   ├── metrics.py                 # Classification (12) & regression (10) metrics
│   │   ├── ranking.py                 # Elo-style tournament ranking
│   │   ├── auto_config.py             # Auto-detect primary metric from dataset
│   │   ├── active_learning.py         # Uncertainty-based sample recommendations
│   │   └── types.py                   # EvalConfig, ModelResult
│   │
│   ├── search/                        # Literature search
│   │   ├── orchestrator.py            # Coordinates Semantic Scholar + PubMed + Tavily
│   │   ├── semantic_scholar.py        # Semantic Scholar API client
│   │   ├── pubmed.py                  # PubMed API client
│   │   ├── tavily.py                  # Tavily web search client
│   │   └── types.py                   # SearchResult types
│   │
│   ├── llm/                           # LLM integration
│   │   ├── client.py                  # Claude API wrapper
│   │   ├── cost.py                    # Token & cost tracking
│   │   ├── prompts.py                 # System prompts for each agent
│   │   └── parser.py                  # Thought/Action response parsing
│   │
│   ├── report/                        # Report generation
│   │   ├── generator.py               # Markdown report rendering
│   │   ├── reviewer.py                # Automated peer review
│   │   ├── summary_pdf.py             # One-page visual summary PDF
│   │   └── template.md.jinja          # Jinja2 report template
│   │
│   ├── viz/                           # Visualization
│   │   ├── profiling.py               # Dataset profile charts
│   │   ├── preprocessing.py           # Before/after distributions, PCA
│   │   ├── training.py                # Model comparison, confusion matrices
│   │   └── architecture.py            # Architecture diagrams
│   │
│   ├── export/                        # Model export
│   │   └── exporter.py                # Generates standalone train.py/predict.py
│   │
│   ├── checkpoint.py                  # Resume from checkpoint
│   ├── memory.py                      # Cross-run memory (HP priors)
│   ├── complexity.py                  # Adaptive complexity scoring
│   ├── guardrails.py                  # Validation & auto-repair
│   ├── resilience.py                  # Timeout & fallback handling
│   ├── iteration.py                   # Iteration loop logic
│   ├── batch.py                       # Batch dataset processing
│   ├── experiment_log.py              # JSONL experiment logging
│   ├── validation.py                  # Step-level validation
│   ├── live_dashboard.py              # Real-time terminal dashboard
│   └── dashboard.py                   # Dashboard utilities
│
├── tests/                             # Test suite
│   ├── conftest.py                    # Shared fixtures
│   ├── test_config.py                 # Configuration tests
│   ├── test_evaluation.py             # Metric computation tests
│   ├── test_registry.py               # Model registry tests
│   ├── test_report.py                 # Report generation tests
│   └── test_custom_model.py           # Custom model design tests
│
├── docs/                              # Step-by-step implementation documentation
│   ├── implementation_phases.md       # Overview of all implementation phases
│   ├── challenges_and_learnings.md    # Design decisions & lessons learned
│   ├── gpu_docs/                      # GPU/foundation model documentation
│   │   ├── overview.md
│   │   ├── phase1_frozen_embeddings.md
│   │   ├── phase2_finetuning.md
│   │   └── phase3_hybrid_integration.md
│   └── step0_project_skeleton.md      # Steps 0-32: each implementation step
│   └── step1_data_loading_profiling.md     documented individually
│   └── ...                                 (32 step files total)
│
├── test_all_datasets.py               # QC: run all 11 GenBio datasets
├── pyproject.toml                     # Package definition & dependencies
├── Dockerfile                         # CPU Docker image
├── Dockerfile.gpu                     # GPU Docker image (+ AIDO models)
└── .dockerignore
```

## Key Entry Points

| What you want to do | Where to look |
|---------------------|---------------|
| Run the pipeline | `co_scientist/cli.py` |
| Understand the agent system | `co_scientist/agents/react.py`, `coordinator.py`, `debate.py` |
| See how models are trained | `co_scientist/modeling/trainer.py`, `registry.py` |
| See how data is preprocessed | `co_scientist/data/preprocess.py` |
| Understand AIDO integration | `co_scientist/modeling/foundation.py`, `aido_finetune.py` |
| See how reports are generated | `co_scientist/report/generator.py`, `template.md.jinja` |
| Read the full architecture | `ARCHITECTURE.md` |
| Read the research paper | `research_paper.md` |

## Documentation Map

| File | What it covers |
|------|----------------|
| `README.md` | Quick start, overview, how it works |
| `GUIDE.md` | Docker setup, GPU/Colab instructions, CLI reference, environment variables, testing |
| `ARCHITECTURE.md` | Full system design: agents, pipeline, data layer, modeling, evaluation, resilience |
| `research_paper.md` | Research paper with methods, results (macro F1=0.97, Spearman=0.69), limitations |
| `docs/implementation_phases.md` | Roadmap of all 32 implementation steps |
| `docs/step*.md` | Individual step docs (one per feature/module built) |
| `docs/gpu_docs/` | Foundation model integration: frozen embeddings, fine-tuning, hybrid features |
| `docs/challenges_and_learnings.md` | Design tradeoffs, what worked, what didn't |
