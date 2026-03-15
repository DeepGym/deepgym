"""Read input.csv, filter rows where age > 30, write to output.csv."""

import csv


def main():
    with open('input.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    filtered = [row for row in rows if int(row['age']) > 30]

    with open('output.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'age', 'city'])
        writer.writeheader()
        writer.writerows(filtered)

    print(len(filtered))


if __name__ == '__main__':
    main()
