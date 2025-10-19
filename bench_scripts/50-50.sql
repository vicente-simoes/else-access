-- simple mix: one update and one select per transaction
\set aid1 int(random(1, 1000000))
\set aid2 int(random(1, 1000000))
BEGIN;
UPDATE pgbench_accounts SET abalance = abalance + 1 WHERE aid = :aid1;
SELECT abalance FROM pgbench_accounts WHERE aid = :aid2;
COMMIT;
