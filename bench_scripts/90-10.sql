-- one simple read-only query
\set aid int(1 + random() * 1000000)
SELECT abalance FROM pgbench_accounts WHERE aid = :aid;

