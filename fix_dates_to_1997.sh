#!/bin/bash
echo "============================================"
echo "Fixing Northwind Database Dates to 1997"
echo "============================================"

echo "1. Backing up original database..."
cp data/northwind.sqlite data/northwind.sqlite.backup
echo "   ✓ Backup created: data/northwind.sqlite.backup"

echo ""
echo "2. Current date range:"
sqlite3 data/northwind.sqlite "SELECT '   ', MIN(OrderDate), 'to', MAX(OrderDate) FROM Orders;"

echo ""
echo "3. Shifting all dates back 16 years (2012-2023 → 1996-2007)..."
sqlite3 data/northwind.sqlite "UPDATE Orders SET OrderDate = date(OrderDate, '-16 years');"

echo ""
echo "4. New date range:"
sqlite3 data/northwind.sqlite "SELECT '   ', MIN(OrderDate), 'to', MAX(OrderDate) FROM Orders;"

echo ""
echo "5. Verifying data in required date ranges:"
sqlite3 data/northwind.sqlite << 'SQL'
.mode column
.headers on
SELECT 
    'Summer 1997 (Jun)' as Period,
    COUNT(*) as Orders,
    MIN(OrderDate) as Start,
    MAX(OrderDate) as End
FROM Orders 
WHERE OrderDate BETWEEN '1997-06-01' AND '1997-06-30'
UNION ALL
SELECT 
    'Winter 1997 (Dec)' as Period,
    COUNT(*) as Orders,
    MIN(OrderDate) as Start,
    MAX(OrderDate) as End
FROM Orders 
WHERE OrderDate BETWEEN '1997-12-01' AND '1997-12-31';
SQL

echo ""
echo "============================================"
echo "✓ Database dates fixed successfully!"
echo "============================================"
