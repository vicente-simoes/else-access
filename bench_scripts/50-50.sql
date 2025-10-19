-- simple mix: one update and one select per transaction
\set aid1 (1 + random() * 1000000)::int
\set aid2 (1 + random() * 1000000)::int
BEGIN;
UPDATE pgbench_accounts SET abalance = abalance + 1 WHERE aid = :aid1;
SELECT abalance FROM pgbench_accounts WHERE aid = :aid2;
COMMIT;
