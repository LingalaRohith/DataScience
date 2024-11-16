package scalation
package modeling
package forecasting2

import scala.math.min
import scala.runtime.ScalaRunTime.stringOf

import scalation.mathstat._

object CovidData:

  import scala.collection.mutable.HashMap

  val fileName = "covid_19_weekly.csv"

  val header = Array ("new_cases",
    "new_deaths",
    "reproduction_rate",
    "icu_patients",
    "hosp_patients",
    "new_tests",
    "positive_rate",
    "tests_per_case",
    "people_vaccinated",
    "people_fully_vaccinated",
    "total_boosters",
    "new_vaccinations",
    "excess_mortality_cumulative_absolute",
    "excess_mortality_cumulative",
    "excess_mortality",
    "excess_mortality_cumulative_per_million")

  val response = "new_deaths"                                   // main response/output variable
  val NO_EXO   = Array.ofDim [String] (0)                       // empty array => no exogenous variables

  val yy = Example_Covid.loadData_y ()
  //  val y  = yy                                                   // full
  val y  = yy(0 until 116)                                      // clip the flat end
  //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /** Load the Covid-19 weekly data into a matrix for the exogenous variables x
   *  and a vector for the response/endogenous variable y.
   *  @param x_strs  the column names for the exogenous variables x
   *  @param y_str   the column name for the endogenous variable y
   *  @param trim    the number of initial rows to trim away (e.g., they are all 0)
   */
  def loadData (x_strs: Array [String], y_str: String = response, trim: Int = 0): (MatrixD, VectorD) =
    val col = HashMap [String, Int] ()
    for i <- header.indices do col += header(i) -> i

    val data = MatrixD.load (fileName, 1+trim, 1)             // skip first row (header) + trim first column
    val x_cols = for s <- x_strs yield col(s)
    (data(?, x_cols), data(?, col(y_str)))
  end loadData

  //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /** Load the Covid-19 weekly data into a vector for the response/endogenous variable y.
   *  @param y_str  the column name for the endogenous variable y
   *  @param trim   the number of initial rows to trim away (e.g., they are all 0)
   */
  def loadData_y (y_str: String = response, trim: Int = 0): VectorD =
    val col = HashMap [String, Int] ()
    for i <- header.indices do col += header(i) -> i

    val data = MatrixD.load (fileName, 1+trim, 1)             // skip first row (header) + trim first column
    data(?, col(y_str))
  end loadData_y

  //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
  /** Load the Covid-19 weekly data into a matrix for the variables y.
   *  @param y_str  the column names for the variables y (e.g., used in a VAR model)
   *  @param trim   the number of initial rows to trim away (e.g., they are all 0)
   */
  def loadData_yy (y_strs: Array [String], trim: Int = 0): MatrixD =
    val col = HashMap [String, Int] ()
    for i <- header.indices do col += header(i) -> i

    val data = MatrixD.load (fileName, 1+trim, 1)             // skip first row (header) + trim first column
    val y_cols = for s <- y_strs yield col(s)
    data(?, y_cols)
  end loadData_yy

end CovidData

import CovidData._

@main def RWmodel (): Unit =
  val hh = 6
  val mod = new RandomWalk(y, hh)
  mod.trainNtest ()()

  banner ("In-ST Test: Random Walk Model")
  mod.forecastAll (y)
  mod.diagnoseAll (y, mod.getYf)                                        // should agree with evalForecasts
  Forecaster.evalForecasts (mod, mod.getYb, hh)

  banner ("TnT Test: Random Walk Model")
  mod.setSkip (0)
  mod.rollValidate ()                                                 // TnT with Rolling Validation
  mod.diagnoseAll (y, mod.getYf, Forecaster.teRng (y.dim))            // only diagnose on the testing set

end RWmodel


@main def AR1 (): Unit =
  val hh = 6
  val hp = AR.hp
  hp("p") = 1
  val mod = new AR(y, hh)
  mod.trainNtest()()

  banner ("In-ST Test: Auto-Regressive AR(1) Model")
  mod.forecastAll (y)
  mod.diagnoseAll (y, mod.getYf)                                        // should agree with evalForecasts
  Forecaster.evalForecasts (mod, mod.getYb, hh)

  banner ("TnT Test: Auto-Regressive AR(1) Model")
  mod.setSkip (0)
  mod.rollValidate ()                                                 // TnT with Rolling Validation
  mod.diagnoseAll (y, mod.getYf, Forecaster.teRng (y.dim))            // only diagnose on the testing set

end AR1

@main def AR2 (): Unit =
  val hh = 6
  val hp = AR.hp
  hp("p") = 2
  val mod = new AR(y, hh)
  mod.trainNtest()()

  banner ("In-ST Test: Auto-Regressive AR(2) Model")
  mod.forecastAll (y)
  mod.diagnoseAll (y, mod.getYf)                                        // should agree with evalForecasts
  Forecaster.evalForecasts (mod, mod.getYb, hh)

  banner ("TnT Test: Auto-Regressive AR(2) Model")
  mod.setSkip (0)
  mod.rollValidate ()                                                 // TnT with Rolling Validation
  mod.diagnoseAll (y, mod.getYf, Forecaster.teRng (y.dim))            // only diagnose on the testing set

end AR2