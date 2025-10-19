-- one simple read-only query
\set aid int(random(1, 1000000))
SELECT abalance FROM pgbench_accounts WHERE aid = :aid;
