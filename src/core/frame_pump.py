"""Единственная точка, где крутится taskMgr Panda, с защитой от рекурсии.

ЗАЧЕМ
-----
Кадры Panda в UI гонит QTimer -> taskMgr.step(). Датасетный цикл при этом
внутри себя звал QApplication.processEvents(), чтобы окно не «замерзало», —
а processEvents доставляет тик того же QTimer, который снова входит в
taskMgr.step(). Panda обнаруживает повторный вход и печатает

    :task(warning): Ignoring recursive poll() within another task

ПРОПУСКАЯ кадр. Последствия ровно те, что наблюдались: сцена перестаёт
обновляться («ничего не рендерится»), счётчик settle-кадров врёт (мы думаем,
что прогнали 60 кадров, а реально — часть проигнорирована), и снимок
рассинхронизируется с маской.

РЕШЕНИЕ
-------
Все, кому нужны кадры (QTimer, settle-циклы датасета, захват), ходят через
FramePump.step(). Он:

  * держит флаг _in_step и при повторном входе НЕ зовёт taskMgr.step()
    повторно, а просто возвращает управление — рекурсивного poll() больше
    не возникает в принципе;
  * считает реально выполненные кадры, поэтому «прогнать N кадров» означает
    N ВЫПОЛНЕННЫХ кадров, а не N попыток.

Это не «затыкание предупреждения»: повторный вход в шаг задач физически
некорректен, и подавлять надо именно его, а не сообщение.
"""


class FramePump:
    """Защищённый насос кадров.

    ВАЖНО: недостаточно просто «ходить через насос» — в проекте (и в самой
    Panda/direct) есть код, зовущий taskMgr.step() напрямую, и любой такой
    вызов из-под уже идущего кадра снова даёт рекурсивный poll. Поэтому насос
    ПОДМЕНЯЕТ сам taskMgr.step своей защищённой версией: тогда безопасны ВСЕ
    вызывающие, включая те, которых мы не знаем.
    """

    def __init__(self, base):
        self.base = base
        self._in_step = False
        self.executed = 0        # сколько кадров реально выполнено
        self.suppressed = 0      # сколько повторных входов отсечено

        # Подменяем taskMgr.step на защищённую обёртку. Оригинал держим при
        # себе — только его и зовём, уже под флагом.
        tm = getattr(base, "taskMgr", None)
        self._orig_step = getattr(tm, "step", None) if tm is not None else None
        if self._orig_step is not None:
            tm.step = self._guarded_step

    # ------------------------------------------------------------------
    def _raw_step(self):
        if self._orig_step is not None:
            self._orig_step()
        else:
            self.base.graphicsEngine.render_frame()

    def _guarded_step(self):
        """Замена taskMgr.step: один кадр, но никогда рекурсивно."""
        if self._in_step:
            self.suppressed += 1
            return
        self._in_step = True
        try:
            self._raw_step()
            self.executed += 1
        finally:
            self._in_step = False

    # ------------------------------------------------------------------
    def step(self, count=1):
        """Выполнить `count` кадров. Возвращает число ВЫПОЛНЕННЫХ кадров.

        При повторном входе (нас позвали из кода, который сам крутится внутри
        кадра) возвращает 0 и ничего не делает.
        """
        if self._in_step:
            self.suppressed += 1
            return 0

        done = 0
        self._in_step = True
        try:
            for _ in range(max(1, int(count))):
                self._raw_step()
                done += 1
                self.executed += 1
        finally:
            self._in_step = False
        return done

    # ------------------------------------------------------------------
    @property
    def busy(self):
        """True, если кадр уже выполняется (мы внутри step)."""
        return self._in_step

    def stats(self):
        return {"executed": self.executed, "suppressed": self.suppressed}
