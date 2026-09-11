# Execution: paper-only broker contract

- Broker protocol: submit(intent)->ExecutionResult, close_position, account, positions.
- AlpacaPaperBroker: asserts PAPER at init;意 idempotent intent_ids; verifies filled_qty/avg price;
  PARTIAL flagged explicitly; REJECTED/FAILED recorded with message.
- submit_order()==fill is NEVER assumed. Local portfolio reconciles vs broker positions.
- Live money: no code path. Requesting mode!=paper raises PaperOnlyError / SystemExit.
