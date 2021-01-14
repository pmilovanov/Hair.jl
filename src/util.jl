using Base.Threads: @spawn

# From https://github.com/JuliaLang/julia/issues/7626

"Like @async except it prints errors to the terminal."
macro asynclog(expr)
  quote
    @async try
      $(esc(expr))
    catch ex
      bt = stacktrace(catch_backtrace())
      showerror(stderr, ex, bt)
      rethrow(ex)
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
      showerror(stderr, ex, bt)
      rethrow(ex)
    end
  end
end
