import utilities

def run():
    currency_data = utilities.get_dataset()
    print(currency_data.head())


if __name__ == "__main__":
    run()