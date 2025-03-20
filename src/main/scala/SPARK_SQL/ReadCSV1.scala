package SPARK_SQL
import  org.apache.spark.sql.SparkSession

object ReadCSV1 {
  def main(args: Array[String]): Unit = {
    val spark=SparkSession
      .builder()
      .appName("ReadCSV1")
      .master("local[1]")
      .getOrCreate()
    import spark.implicits._
    val datadf=spark.read
      .option("header",value=true)
        //|//
      //.option("delimiter","|")
      //;//
     //.option("lineSep",";")
      // one_file_data
      //.csv(path = "data/data/read2.csv")
      //two_file_data
     .csv(paths=List("data/data/JSON_Files/read.csv","data/data/JSON_Files/read2.csv"):_*)
      //.csv("data/data/JSON_Files/*.csv")
    datadf.printSchema()
    datadf.show()
  }
}
