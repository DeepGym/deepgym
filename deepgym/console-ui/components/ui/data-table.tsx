'use client';

import {
  type ColumnDef,
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  type RowData,
  type SortingState,
  useReactTable,
} from '@tanstack/react-table';
import { type ReactNode, useState } from 'react';
import { Pagination } from '@/components/base/pagination/pagination';
import {
  Table,
  TableBody,
  TableCell,
  TableColumn,
  TableHeader,
  TableRow,
} from '@/components/base/table/table';
import { ChevronSortDown } from '@/components/foundations/icons/chevrons';
import { cx } from '@/utils/cx';

declare module '@tanstack/react-table' {
  interface ColumnMeta<TData extends RowData, TValue> {
    align?: 'start' | 'end';
    /** Classes for this column's cells (width, font). */
    className?: string;
  }
}

/** Which page is shown, out of how many, and how to move to another. */
interface Paging {
  pageIndex: number;
  pageCount: number;
  onPageChange: (pageIndex: number) => void;
}

interface DataTableProps<T extends { id: string }> {
  label: string;
  data: T[];
  columns: ColumnDef<T>[];
  /** Enter or click on a row. */
  onRowAction?: (row: T) => void;
  initialSort?: SortingState;
  /** Rows per page; omit to show every row. */
  pageSize?: number;
  /**
   * Pages fetched from the server, instead of `pageSize`: `data` is the page at `pageIndex`.
   * Sorting is off, since it could only reorder that one page.
   */
  serverPaging?: Paging;
  empty?: ReactNode;
}

/**
 * A TanStack table model (sorting, paging) rendered with BoardUI's React Aria table:
 * arrow keys move between rows, Enter runs the row action, headers sort.
 */
export function DataTable<T extends { id: string }>({
  label,
  data,
  columns,
  onRowAction,
  initialSort = [],
  pageSize,
  serverPaging,
  empty,
}: DataTableProps<T>) {
  const [sorting, setSorting] = useState<SortingState>(initialSort);
  const table = useReactTable({
    data,
    columns,
    state: { sorting },
    onSortingChange: setSorting,
    enableSorting: !serverPaging,
    getRowId: (row) => row.id,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    ...(pageSize
      ? { getPaginationRowModel: getPaginationRowModel(), initialState: { pagination: { pageSize } } }
      : {}),
    autoResetPageIndex: true,
  });
  const sort = sorting[0];
  const paging: Paging = serverPaging ?? {
    pageIndex: table.getState().pagination.pageIndex,
    pageCount: table.getPageCount(),
    onPageChange: (pageIndex) => table.setPageIndex(pageIndex),
  };

  return (
    <div>
      <Table
        aria-label={label}
        size="sm"
        sortDescriptor={
          sort ? { column: sort.id, direction: sort.desc ? 'descending' : 'ascending' } : undefined
        }
        onSortChange={(next) =>
          setSorting([{ id: String(next.column), desc: next.direction === 'descending' }])
        }
        onRowAction={onRowAction ? (key) => onRowAction(table.getRow(String(key)).original) : undefined}
      >
        <TableHeader>
          {table.getFlatHeaders().map((header) => {
            const align = header.column.columnDef.meta?.align;
            return (
              <TableColumn
                key={header.id}
                id={header.id}
                isRowHeader={header.index === 0}
                allowsSorting={header.column.getCanSort()}
                className={cx('group', align === 'end' && 'text-end')}
              >
                {({ allowsSorting, sortDirection }) => (
                  <span
                    className={cx('inline-flex items-center gap-0.5', align === 'end' && 'flex-row-reverse')}
                  >
                    {flexRender(header.column.columnDef.header, header.getContext())}
                    {allowsSorting && (
                      <ChevronSortDown
                        className={cx(
                          'size-4 text-foreground-icon-tertiary transition-[transform,opacity]',
                          !sortDirection && 'opacity-0 group-hover:opacity-70',
                          sortDirection === 'ascending' && 'rotate-180',
                        )}
                      />
                    )}
                  </span>
                )}
              </TableColumn>
            );
          })}
        </TableHeader>
        <TableBody items={table.getRowModel().rows} renderEmptyState={() => empty}>
          {(row) => (
            <TableRow id={row.id}>
              {row.getVisibleCells().map((cell) => (
                <TableCell
                  key={cell.id}
                  className={cx(
                    cell.column.columnDef.meta?.align === 'end' && 'text-end',
                    cell.column.columnDef.meta?.className,
                  )}
                >
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          )}
        </TableBody>
      </Table>
      {paging.pageCount > 1 && (
        <div className="border-t border-separator-border px-3 py-2.5">
          <Pagination
            page={paging.pageIndex + 1}
            totalPages={paging.pageCount}
            onChange={(page) => paging.onPageChange(page - 1)}
          />
        </div>
      )}
    </div>
  );
}
