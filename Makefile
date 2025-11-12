PROJECT_DIR?=projects/secure-rag-soc

.PHONY: run ingest test clean

run:
	$(MAKE) -C $(PROJECT_DIR) run

ingest:
	$(MAKE) -C $(PROJECT_DIR) ingest

test:
	$(MAKE) -C $(PROJECT_DIR) test

clean:
	$(MAKE) -C $(PROJECT_DIR) clean
