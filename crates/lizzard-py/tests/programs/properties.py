# lizzard: module
# Properties with @x.getter, @x.setter and @x.deleter, and del of attributes.
# Its output must match plain CPython's.


class Account:
    def __init__(self, owner: str, cents: int) -> None:
        self.owner: str = owner
        self._cents: int = cents
        self.note: str | None = "new"
        self.log: list[str] = []

    @property
    def balance(self) -> float:
        return self._cents / 100

    @balance.setter
    def balance(self, value: float) -> None:
        if value < 0:
            raise ValueError("negative balance")
        self.log.append(f"set {value}")
        self._cents = round(value * 100)

    @balance.deleter
    def balance(self) -> None:
        self.log.append("closed")
        self._cents = 0

    @property
    def label(self) -> str:
        return "plain"

    @label.getter
    def label(self) -> str:
        return f"{self.owner}: {self.balance:.2f}"


class Celsius:
    def __init__(self) -> None:
        self._t: float = 0.0

    @property
    def temp(self) -> float:
        return self._t

    @temp.setter
    def temp(self, v: float) -> None:
        self._t = v

    @property
    def fahrenheit(self) -> float:
        return self._t * 9 / 5 + 32

    @fahrenheit.setter
    def fahrenheit(self, f: float) -> None:
        self.temp = (f - 32) * 5 / 9


def main() -> None:
    a = Account("ann", 1050)
    print(a.balance, a.label)
    a.balance = 20.25
    a.balance += 1.0
    print(a.balance, a.label, a.log)
    try:
        a.balance = -1.0
    except ValueError as e:
        print("rejected:", e)
    del a.balance
    print(a.balance, a.log)
    del a.note
    try:
        print(a.note)
    except AttributeError as e:
        print("AttributeError:", e)
    c = Celsius()
    c.fahrenheit = 212.0
    print(c.temp, c.fahrenheit)


if __name__ == "__main__":
    main()
