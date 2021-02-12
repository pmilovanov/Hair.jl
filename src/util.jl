using Base.Threads: @spawn

# From https://github.com/JuliaLang/julia/issues/7626


SHOWERROR_LOCK = ReentrantLock()


"Like @async except it prints errors to the terminal."
macro asynclog(expr)
  quote
    @async try
      $(esc(expr))
    catch ex
      bt = stacktrace(catch_backtrace())
      lock(SHOWERROR_LOCK) do
        showerror(stderr, ex, bt)
        println()
      end
      rethrow(ex)
    end
  end
end

macro asynclog(channelexpr, expr)
  quote
    @async try
      $(esc(expr))
    catch ex
      bt = stacktrace(catch_backtrace())
      lock(SHOWERROR_LOCK) do
        showerror(stderr, ex, bt)
        println()
      end
      rethrow(ex)
    finally
      close($(esc(channelexpr)))
    end
  end
end



"Like @spawn except it prints errors to the terminal."
macro spawnlog(expr)
  quote
    @spawn try
      $(esc(expr))
    catch ex
      bt = stacktrace(catch_backtrace())
      lock(SHOWERROR_LOCK) do
        showerror(stderr, ex, bt)
        println()
      end
      rethrow(ex)
    end
  end
end

macro spawnlog(channelexpr, expr)
  quote
    @spawn try
      $(esc(expr))
    catch ex
      bt = stacktrace(catch_backtrace())
      lock(SHOWERROR_LOCK) do
        showerror(stderr, ex, bt)
        println()
      end
      rethrow(ex)
    finally
      close($(esc(channelexpr)))
    end
  end
end
