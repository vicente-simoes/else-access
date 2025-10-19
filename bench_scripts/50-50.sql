-- simple mix: one update and one select per transaction
BEGIN;
UPDATE pgbench_accounts SET abalance = abalance + 1 WHERE aid = random_between(1, 1000000);
SELECT abalance FROM pgbench_accounts WHERE aid = random_between(1, 1000000);
COMMIT;