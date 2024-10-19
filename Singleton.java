public class Singleton {
    private static volatile Singleton instance;
    private String data;
    private Singleton(String data) {
        this.data= data;
    }
    public static Singleton getInstance(String data) {
        Singleton result = instance;
        if (result == null) {
            synchronized (Singleton.class) {
                result = instance;
                if (result == null) {
                    instance = result = new Singleton (data);
                }
            }
        }
        return result;
    }

    public String getData() {
        return data;
    }

        public static void main(String[] args) {
            // Create the first singleton instance
            Singleton firstInstance = Singleton.getInstance("First");
    
            // Create the second singleton instance
            Singleton secondInstance = Singleton.getInstance("Second");
    
            // Both instances should refer to the same object, so the data will be the same
            System.out.println("First Instance Data: " + firstInstance.getData());
            System.out.println("Second Instance Data: " + secondInstance.getData());
    
            // Check if both instances are the same
            System.out.println("Are both instances the same? " + (firstInstance == secondInstance));
        }
    
}
